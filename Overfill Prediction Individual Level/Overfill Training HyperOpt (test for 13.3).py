# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

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

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from databricks import feature_store
from databricks.feature_store import feature_table
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup

from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import LongType, DecimalType, FloatType
import mlflow
from mlflow.models.signature import infer_signature



from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql import SparkSession
from math import exp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score
# from imblearn.under_sampling import RandomUnderSampler

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

import mlflow
# from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient


# COMMAND ----------

start_date = '2023-01-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
# end_date = '2024-01-01'

sdf = jobs_query(start_date,end_date)
display(sdf)


# COMMAND ----------

sdf = sdf.filter((sdf.NEEDED >0)&(sdf.COMPANY_ORIGIN == 'BC')&(sdf.JOB_NEEDED_ORIGINAL_COUNT>0)&(sdf.POSTING_LEAD_TIME_DAYS>0))

sdf = sdf.withColumn('Work', F.when(sdf.target_var == 'Worked', 1).otherwise(0))
display(sdf)

# COMMAND ----------

sdf = optimize_spark(sdf)

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
        # feature_names=['JOBS_WORKED_TOTAL', 'JOBS_WORKED_LAST_30_DAYS', 'JOBS_WORKED_LAST_90_DAYS', 'SHIFTS_WORKED_TOTAL', 'SHIFTS_WORKED_LAST_7_DAYS', 'SHIFTS_WORKED_LAST_30_DAYS', 'SHIFTS_WORKED_LAST_90_DAYS', 'TOTAL_HOURS_WORKED_TOTAL', 'TOTAL_HOURS_WORKED_LAST_7_DAYS', 'TOTAL_HOURS_WORKED_LAST_30_DAYS', 'TOTAL_HOURS_WORKED_LAST_90_DAYS','AVG_WAGE_LAST_7_DAYS','AVG_WAGE_LAST_30_DAYS','AVG_WAGE_LAST_90_DAYS','AVG_WAGE_LAST_365_DAYS'],
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
    label='Work'
)

training_pd = training_set.load_df()
# display(training_pd)

# COMMAND ----------

df = optimize_spark(training_pd).toPandas()

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing and training preparation

# COMMAND ----------

# Defines the columns that correspond to a job and creates final dataset to split for training and testing.
cols_to_drop = ['JOB_STATUS_ENUM', 'JOB_STATUS', 'JOB_OVERFILL', 'INVITED_WORKER_COUNT', 'SEGMENT_INDEX', 'NEEDED', 'POSITION_ID', 'COMPANY_ID', 'SCHEDULE_ID', 'JOB_ID','USER_ID', 'JOB_CITY','COUNTY','target_var', 'application_status', 'successful_application_count', 'max_successful_app_start', 'min_successful_app_end', 'max_successful_app_end', 'job_schedule', 'running_schedule_array', 'JOB_TITLE', 'JOB_STATE']
# df2 = df[(df['target_var']!='Early Cancel')&(df['JOB_STATUS']!='Cancelled')]
df2 = df.copy()
df2['Wage_diff_7']=df2['AVG_WAGE_LAST_7_DAYS']-df2['JOB_WAGE']
df2['Unique_ID']= np.arange(df2.shape[0])
df3 = df2[['Unique_ID', 'USER_ID', 'JOB_ID']]
df4 = df2.drop(columns=cols_to_drop)
df5 = df4.set_index('Unique_ID')
# df5['Past_Work']= df5['cosine_sim'].apply(lambda x: 1 if x==x else 0)

df5 = df5[df5['apply_lead_time_hours']!=0]
df5.info()

# COMMAND ----------

# df6 = spark.createDataFrame(df5)

# write_spark_table_to_databricks_schema(df6, 'overfill_individual_training_data', 'bluecrew.ml', mode = 'overwrite')

# COMMAND ----------

mlflow.autolog()
mlflow.set_registry_uri('databricks-uc')
mlflow.set_tracking_uri("databricks")
# mlflow.set_experiment("antwoineflowers/databricks-uc/jobmatchscoring")
model_name = 'overfill_test'

 # Separate features and target variable
X = df5.drop('Work', axis=1)
y = df5['Work']

# Create Preprocessing Pipeline

# Identify numerical and categorical columns (excluding the label column)
numerical_cols = df5.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.drop('Work')

categorical_cols = df5.select_dtypes(include=['object', 'category']).columns

# Preprocessing for numerical data: impute nulls with 0 and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute with the most frequent and apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# from imblearn.under_sampling import RandomUnderSampler
# under_sampler = RandomUnderSampler(sampling_strategy='not minority')
# X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Split data into training, validation, and test sets
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=9986)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=72688)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the hyperparameter space
space = {
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
}

# Objective function for hyperparameter optimization
def objective(params):
    with mlflow.start_run(run_name='overfill_training', nested=True): 
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Initialize XGBClassifier
        model = XGBClassifier(**params 
        )

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Predict and evaluate using the probabilities of the positive class
        predictions_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate metrics using probabilities
        auc_score = roc_auc_score(y_val, predictions_proba)
        precision, recall, _ = precision_recall_curve(y_val, predictions_proba)
        f1 = f1_score(y_val, (predictions_proba > 0.5).astype(int))
        
        # Convert probabilities to binary outcomes based on a 0.5 cutoff
        predictions_binary = (predictions_proba > 0.5).astype(int)

        return {'loss': -auc_score, 'status': STATUS_OK, 'model': pipeline, 'precision': precision, 'f1_score': f1, 'recall':recall}

# Hyperparameter optimization with Hyperopt
with mlflow.start_run(run_name='Hyperparameter Optimization') as parent_run:
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,  
        algo=tpe.suggest,
        max_evals=10,  
        trials=trials
    )

# Fetch the details of the best run
best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
best_params = space_eval(space, best)

# Print out the best parameters and corresponding loss
print(f"Best parameters: {best_params}")
print(f"Best eval auc: {best_run['loss']}")


# COMMAND ----------

mlflow.sklearn.autolog()
# Log the best hyperparameters
with mlflow.start_run(run_name='best_model_run'):
  model = objective(best_params)
  mlflow.set_tag("mlflow.best_model_run", "best_run")
final_model = model['model']  
auc_roc = -1*model['loss']



# COMMAND ----------

# Define the model name for the registry
registry_model_name = "bluecrew.ml.Overfill_Test"

client = MlflowClient()
client.delete_registered_model(name=registry_model_name)

latest_experiment = find_latest_experiment()
best_run_id = find_best_run_id(latest_experiment, "metrics.training_roc_auc")
model_details = register_run_as_model(registry_model_name, best_run_id)
# update_model_stage(model_details, 'Staging')

# COMMAND ----------

#this doesn't come with predict_proba.  I think you could build a custom model wrapper to fix this, but I didn't.
model = mlflow.pyfunc.load_model(model_uri=f"models:/{registry_model_name}/1")
#this allows you to get the predict_proba method
model2 = mlflow.sklearn.load_model(model_uri=f"models:/{registry_model_name}/1")


pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{registry_model_name}/1")

# COMMAND ----------

predictions = pd.Series(model2.predict_proba(X_val)[:,1])

# COMMAND ----------

X = df5.drop('Work', axis=1)
y = df5['Work']
display(X)

# COMMAND ----------

predictions = pd.Series(final_model.predict_proba(X_val)[:,1])
# predictions = pd.Series(final_model.predict(X_val))
display(predictions)

# COMMAND ----------

# y2 = df5['Work'].reset_index()
# Creating a dictionary by passing Series objects as values
# y2['Predictions']=predictions
x2 = pd.concat([X_val.reset_index(),predictions,y_val.reset_index()['Work']],axis = 1)

# y3 = y2.merge(X.reset_index(), on = 'Unique_ID',how = 'inner')
display(x2)

 

# COMMAND ----------

jobs_and_users = x2.merge(df3, on = 'Unique_ID', how = 'left')
job_modeling = jobs_and_users[['JOB_ID',0,'Work']].groupby('JOB_ID').agg({'Work': 'sum', 'JOB_ID':'count', 0: 'sum'})
job_modeling['work_rate']= job_modeling['Work']/job_modeling['JOB_ID']
job_modeling['predicted_work_rate']= job_modeling[0]/job_modeling['JOB_ID']

# display(job_modeling)

# COMMAND ----------

(rmse, mae, r2, mape) = eval_metrics(job_modeling['Work'], job_modeling[0])

# Print out model metrics
print("Comparing expected show ups:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

(rmse, mae, r2, mape) = eval_metrics(job_modeling['work_rate'], job_modeling['predicted_work_rate'])

# Print out model metrics
print("Comparing Work Rate:")
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)
print("  MAPE: %s" % mape)

# COMMAND ----------

feature_importance = FeatureImportance(final_model)
feature_importance.plot(top_n_features=100)

# COMMAND ----------



# COMMAND ----------


