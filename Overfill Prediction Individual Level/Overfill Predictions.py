# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

# COMMAND ----------

# MAGIC %md
# MAGIC #Preprocessing

# COMMAND ----------

# %pip install databricks-feature-engineering
%pip install databricks-feature-engineering==0.2.1a1
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/ml-modeling-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/mlflow-utils
# MAGIC

# COMMAND ----------

# MAGIC %run ./_resources/overfill-utils

# COMMAND ----------

#Imported libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import exp 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import mlflow
from mlflow.models.signature import infer_signature


# COMMAND ----------

now = datetime.now()
start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")

sdf = jobs_query(start_date,end_date)
display(sdf)



# COMMAND ----------

#currently only trust training data for BC accounts where the account was posted in advance of starting and requiring more than 0 people.
sdf = sdf.filter((sdf.NEEDED >0))



# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists bluecrew.ml.overfill_prediction_labels

# COMMAND ----------

from pyspark.sql import DataFrame

def write_spark_table_to_databricks_schema(df: DataFrame, table_name: str, schema_name: str = 'bluecrew.ml', mode: str = 'overwrite'):
    """
    Write a Spark DataFrame to a table within a specific schema in Databricks.

    Parameters:
    - df: The Spark DataFrame to write.
    - table_name: The name of the target table.
    - schema_name: The name of the schema (database) in Databricks. Default is 'bluecrew.ml'.
    - mode: Specifies the behavior when the table already exists. Options include:
      - 'append': Add the data to the existing table.
      - 'overwrite': Overwrite the existing table.
      - 'ignore': Silently ignore this operation if the table already exists.
      - 'error' or 'errorifexists': Throw an exception if the table already exists.
    """
    # Define the full table path including the schema name
    full_table_name = f"{schema_name}.{table_name}"
    
    # Write the DataFrame to the table in the specified schema
    df.write.mode(mode).saveAsTable(full_table_name)

    print(f"DataFrame written to table {full_table_name} in mode '{mode}'.")



write_spark_table_to_databricks_schema(optimize_spark(sdf), 'overfill_prediction_labels', 'bluecrew.ml')


# COMMAND ----------

sdf = spark.read.format("delta").table('bluecrew.ml.overfill_prediction_labels')

# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #This is a feature lookup that demonstrates how to use point in time based lookups for training sets
      FeatureLookup(
        table_name='feature_store.dev.jobs_data',
        feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'SCHEDULE_NAME_UPDATED', "COUNTY", "JOB_ADDRESS_LATITUDE", "JOB_ADDRESS_LONGITUDE"],
        lookup_key="JOB_ID",
        timestamp_lookup_key="min_successful_app_start"),
      FeatureLookup(
        table_name='feature_store.dev.user_hours_worked_calendar2',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"),
      FeatureLookup(
        table_name='feature_store.dev.user_snc_ncns_calendar',
        lookup_key="USER_ID",
        feature_names=['NCNS_SHIFTS_TOTAL', 'NCNS_SHIFTS_LAST_30_DAYS', 'NCNS_SHIFTS_LAST_90_DAYS', 'NCNS_SHIFTS_LAST_365_DAYS', 'SNC_SHIFTS_TOTAL', 'SNC_SHIFTS_LAST_30_DAYS', 'SNC_SHIFTS_LAST_90_DAYS', 'SNC_SHIFTS_LAST_365_DAYS'],
        timestamp_lookup_key="min_successful_app_start"),
      FeatureLookup(
        table_name='feature_store.dev.job_schedule_array',
        lookup_key='JOB_ID',
        timestamp_lookup_key="min_successful_app_start"
      ),
      FeatureLookup(
        table_name='feature_store.dev.user_schedule_array2',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"
      ),
      FeatureLookup(
        table_name='feature_store.dev.user_work_history',
        lookup_key="USER_ID",
        timestamp_lookup_key="min_successful_app_start"
      ), 
      FeatureLookup(
        table_name='feature_store.dev.user_funnel_timeline',
        lookup_key="USER_ID",
        feature_names=['ever_worked'],
        timestamp_lookup_key="min_successful_app_start"
      ), 
      # Calculate a new feature called `cosine_sim` - the cosine similarity between the user's work history and the current job.
      FeatureFunction(
        udf_name='feature_store.dev.cosine_similarity',
        output_name="cosine_sim",
        input_bindings={"arr1":"job_schedule", "arr2": "running_schedule_array"},
      ), 
      # Calculate a new feature called `commute_distance` - the distance between a user's address and the current job address.
      FeatureFunction(
        udf_name='feature_store.dev.distance',
        output_name="commute_distance",
        input_bindings={"lat1":"user_address_latitude", "lon1": "user_address_longitude", "lat2":"JOB_ADDRESS_LATITUDE", "lon2": "JOB_ADDRESS_LONGITUDE"},
      )
]
training_set = fe.create_training_set(
    df = sdf, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=model_feature_lookups,
    exclude_columns=[], # exclude columns as we don't want them as feature
    label=None
)

training_pd = training_set.load_df()
# display(training_pd)

# COMMAND ----------

# Converts the Spark df + features to a pandas DataFrame and turns boolean columns into integers
# convert this to do the type casting first and then use toPandas()?
df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))
df.set_index(['JOB_ID','USER_ID'], inplace = True)

# COMMAND ----------

# df['Wage_diff_7']=df['JOB_WAGE']/df['avg_wage_last_7_days']
df['Wage_diff']=df['JOB_WAGE']/df['avg_wage_total']

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
# Define the model name for the registry
registry_model_name = "bluecrew.ml.overfill_test"

model2 = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")
model_info = mlflow.models.get_model_info(f"models:/{registry_model_name}/1")
sig_dict = model_info._signature_dict
# sig_dict['inputs']




# COMMAND ----------


import ast
true = True
false = False
schema_dict = {}
input_dict_list = eval(sig_dict['inputs'])
# print(input_dict)
for value in input_dict_list:
  # print(value)
  schema_dict[value['name']]=value['type']
schema_dict


# COMMAND ----------

df2 = df.copy()
df2['predicted_prob'] = model2.predict_proba(df2[[key for key in schema_dict]])[:,1]
df2['prediction'] = model2.predict(df2[[key for key in schema_dict]])

# COMMAND ----------

df2.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Need to add the post processing outputs to get to new predictions for Streamlit

# COMMAND ----------

# # Builds DataFrame with predicted and actual values for test set
# # Determines the range of values to consider for upper and lower bounds of a prediction interval.  Want this to be large enough to cover the actual work rate most of the time, but remain small enough to be useful
# interval = .7
# a = y_valid.reset_index()
# a = a.rename(columns={'Work':'Actual_Show_Up_Rate'})
# a['Predicted_Show_Up_Rate'] = preds
# a['Delta'] = abs(a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate'])
# a['Signed_Delta'] = a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate']

# # 50% interval
# a['lowq'] = a['Signed_Delta'].quantile((1-interval)/2)
# a['highq'] = a['Signed_Delta'].quantile(1-(1-interval)/2)

# a['Dataset']="Test"
# print(a)

# COMMAND ----------

# # Builds DataFrame with predicted and actual values for training set
# b = y_train.reset_index()
# b = b.rename(columns={'Work':'Actual_Show_Up_Rate'})
# b['Predicted_Show_Up_Rate'] = my_pipeline.predict(X_train)
# b['Delta'] = abs(b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate'])
# b['Signed_Delta'] = b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate']
# b['lowq'] = b['Signed_Delta'].quantile((1-interval)/2)
# b['highq'] = b['Signed_Delta'].quantile(1-(1-interval)/2)
# b['Dataset']="Training"
# print(b)

# COMMAND ----------

# # Builds DataFrame with predicted and actual values for future set.  Don't have actual values for these.
# d = y_future.reset_index()
# d = d.rename(columns={'Work':'Actual_Show_Up_Rate'})

# d['Predicted_Show_Up_Rate'] = my_pipeline.predict(future_jobs)
# d['Actual_Show_Up_Rate'] = 0
# d['Delta'] = 0
# d['Signed_Delta'] = 0
# #signed delta from test dataset
# d['lowq'] = a['Signed_Delta'].quantile((1-interval)/2)
# d['highq'] = a['Signed_Delta'].quantile(1-(1-interval)/2)
# d['Dataset']="Future"

# d

# COMMAND ----------

# # Merges job data to look at characteristics associated with predictions
# c = pd.concat([a,b,d])

# # creates the upper and lower bounds of the prediction interval
# c['lower_guess']=(c['Predicted_Show_Up_Rate'] + c['lowq']).clip(lower = .01) #can't have negative numbers
# c['upper_guess']=(c['Predicted_Show_Up_Rate'] + c['highq']).clip(upper = 1) #can't have numbers over 1

# #merges with df5 to incorporate job data
# eda_df = c.merge(df5, left_on='JOB_ID', right_index = True, how = 'left').sort_values('JOB_NEEDED_ORIGINAL_COUNT', ascending = True)
# eda_df2 = eda_df[eda_df['Actual_Show_Up_Rate']==0]



# COMMAND ----------

# overfill_added = eda_df.merge(df[['JOB_OVERFILL', 'TOTAL_SUCCESSFUL_SIGN_UPS', 'JOB_ID']], on='JOB_ID', how = 'left')
# overfill_added['JOB_OVERFILL'].fillna(0,inplace=True)
# overfill_added['PAST_PREDICTED_Show_Up_Rate']=overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']+overfill_added['JOB_OVERFILL'])
# # original needed / original needed + overfill
# overfill_added_test = overfill_added[overfill_added['Dataset']=='Test']
# print("Past performance considering all jobs in test data:")

# (rmse, mae, r2, mape) = eval_metrics(overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'Actual_Show_Up_Rate'], overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'PAST_PREDICTED_Show_Up_Rate'])

# # Print out model metrics
# print("  RMSE: %s" % rmse)
# print("  MAE: %s" % mae)
# print("  R2: %s" % r2)

# overfill_added2 = overfill_added_test[overfill_added_test['JOB_OVERFILL']!=0]

# print("Past performance on jobs with overfill added:")
# (rmse, mae, r2, mape) = eval_metrics(overfill_added2['Actual_Show_Up_Rate'], overfill_added2['PAST_PREDICTED_Show_Up_Rate'])

# # Print out model metrics
# print("  RMSE: %s" % rmse)
# print("  MAE: %s" % mae)
# print("  R2: %s" % r2)



# COMMAND ----------

# #calculate costs associated with each prediction
# overfill_added['Overfill_to_100']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Actual_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
# overfill_added['Overfill_to_100_perc'] = 1/overfill_added['Actual_Show_Up_Rate'] - 1
# overfill_added['Overfill_Recommendation']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Predicted_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
# overfill_added['Overfill_rec_perc'] = 1/overfill_added['Predicted_Show_Up_Rate'] - 1
# overfill_added['Overfill_Rec_Upper_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['lower_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
# overfill_added['Overfill_upper_perc'] = 1/overfill_added['lower_guess'] - 1
# overfill_added['Overfill_Rec_Lower_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['upper_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
# overfill_added['Overfill_lower_perc'] = 1/overfill_added['upper_guess'] - 1

# overfill_added['Overfill_Rec_to_Ideal']=overfill_added['Overfill_Recommendation']-overfill_added['Overfill_to_100']
# overfill_added['Overfill_Past_to_Ideal']=overfill_added['JOB_OVERFILL']-overfill_added['Overfill_to_100']

# overfill_added['Overfill_Prediction_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']*-25 if row['Overfill_Rec_to_Ideal']<0 else 0, axis=1).apply(np.round)
# overfill_added['Overfill_Prediction_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else 0, axis=1).apply(np.round)

# overfill_added['Overfill_Past_Rec_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']*-25 if row['Overfill_Past_to_Ideal']<0 else 0, axis=1).apply(np.round)
# overfill_added['Overfill_Past_Rec_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else 0, axis=1).apply(np.round)

# overfill_added['Overfill_Prediction_Cost'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else row['Overfill_Rec_to_Ideal']*-25, axis=1).apply(np.round)
# overfill_added['Overfill_Past_Rec_Cost'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else row['Overfill_Past_to_Ideal']*-25, axis=1).apply(np.round)
# overfill_added[overfill_added['Actual_Show_Up_Rate']>0]
# overfill_added[(overfill_added['Dataset']=='Future')&(overfill_added['JOB_ID']==329636)]

# COMMAND ----------

sdf2 = spark.createDataFrame(df2.reset_index())
sdf2.createOrReplaceTempView('data')

# COMMAND ----------

sdf2.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_INDIVIDUAL_PREDICTIONS').save()


# COMMAND ----------

sns.histplot(data=df2, x='predicted_prob', hue="JOB_TYPE",palette="Spectral",
   alpha=.5, linewidth=0,bins=10, multiple='stack')
plt.show()

# COMMAND ----------

iterations = 10
a = X_valid.copy()
X_valid2 = X_valid.copy()
a['Predicted_Show_Up_Rate'] = my_pipeline.predict(X_valid2)
for i in range(0,iterations):
  X_valid2['JOB_WAGE']=X_valid2['JOB_WAGE']+1
  preds = my_pipeline.predict(X_valid2)
  a['Prediction_'+str(i)] = preds

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))

sns.scatterplot(x="Predicted_Show_Up_Rate",
                    y=f"Prediction_9",
                    data=a)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Original Predicted Show Up Rate')
plt.ylabel('New Predicted Show Up Rate')
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
for i in range(0,iterations):
  sns.scatterplot(x="Predicted_Show_Up_Rate",
                      y=f"Prediction_{i}",
                      size = int(i),
                      hue = int(i),
                      data=a)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Original Predicted Show Up Rate')
plt.ylabel('New Predicted Show Up Rate')
plt.show()

# COMMAND ----------

display(df[df['Dataset']=='Future'])

# COMMAND ----------


