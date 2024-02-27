# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

# COMMAND ----------

# MAGIC %md
# MAGIC #Preprocessing

# COMMAND ----------

# MAGIC %run ./_resources/utils

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
from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import LongType, DecimalType, FloatType
import mlflow
from mlflow.models.signature import infer_signature


# COMMAND ----------

start_date = '2023-01-01'
now = datetime.now()
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

sdf = spark.sql(f"""
/*  Title: JOB_APPLICATION_CANCELLATION_WORKED
    Date: 1/2/2023
    Author: Dan Streeter
    Summary: 
        In order to better plan overfills, we know the proportion of workers that have successfully worked jobs out of the ones that were still in a successful                applied status at the time of job start.  This logic comes from the data team Time to Fill Logic 
    Ticket Information:
    Dependencies:
        BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE
        BLUECREW.MYSQL_BLUECREW.SCHEDULE_WORK_REQUESTS
        BLUECREW.DM.DM_JOB_NEEDED_HISTORY
        BLUECREW.DM.DM_JOBS
        BLUECREW.DM.DM_CM_JOB_APPLIED_HISTORY
        DM.DM_CM_HOURS_WORKED
    Caveats and Notes:
        This doesn't check the people who worked of the applicants.  It only checks total successful applicants and number successfully worked.
    Modifications:

*/

--create or replace table PERSONALIZATION.JOB_APPLICATION_CANCELLATION_WORKED as (
WITH tmp_first_shift AS (
    -- identify the first shift of a job
    SELECT DISTINCT
        tsa.JOB_ID,
        FIRST_VALUE(s.SCHEDULE_ID) OVER (PARTITION BY s.WORK_REQUEST_ID ORDER BY s.CREATED_AT DESC) AS SCHEDULE_ID,
        FIRST_VALUE(tsa.START_TIME) OVER (PARTITION BY tsa.JOB_ID ORDER BY tsa.START_TIME ASC) AS FIRST_SHIFT_START_TIME,
        FIRST_VALUE(tsa.SEGMENT_INDEX) OVER (PARTITION BY tsa.JOB_ID ORDER BY tsa.START_TIME ASC) AS FIRST_SEGMENT
    FROM BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE tsa
    LEFT JOIN BLUECREW.MYSQL_BLUECREW.SCHEDULE_WORK_REQUESTS s
      ON tsa.JOB_ID = s.WORK_REQUEST_ID
    WHERE tsa.ACTIVE = TRUE
        -- add a shift injects shifts into previous jobs with a matching wage and job type
        -- these retroactive shifts should not be included in time to fill calculations
        AND tsa.CREATED_AT < to_TIMESTAMP(tsa.START_TIME)
        --AND tsa._FIVETRAN_DELETED = FALSE
        ), -- account for bug causing duplicate rows in TSA

tmp_first_shift_full AS (
    -- get all columns for first shift and job information
        SELECT
            fs.JOB_ID,
            tsa.SEGMENT_INDEX,
            j.POSITION_ID,
            to_TIMESTAMP(tsa.START_TIME) AS START_TIME,
            TIMESTAMPADD(HOUR, -(tsa.START_TIME_OFFSET/60), to_TIMESTAMP(tsa.START_TIME)) AS START_TIME_LOCAL,
            to_TIMESTAMP(tsa.END_TIME) AS END_TIME,
            TIMESTAMPADD(HOUR, -(tsa.END_TIME_OFFSET/60),to_TIMESTAMP(tsa.END_TIME)) AS END_TIME_LOCAL,
            fs.SCHEDULE_ID,
            j.JOB_CREATED_AT,
            j.JOB_TYPE,
            j.JOB_TITLE,
            j.JOB_WAGE,
            j.JOB_STATUS_ENUM,
            j.JOB_STATUS,
            j.JOB_OVERFILL,
            j.JOB_CITY,
            j.JOB_STATE,
            j.COMPANY_ID,
            j.INVITED_WORKER_COUNT,
            j.JOB_NEEDED_ORIGINAL_COUNT,
            COALESCE(nh.NEEDED, j.JOB_NEEDED_ORIGINAL_COUNT) AS NEEDED,
            j.JOB_SHIFTS
            /*
            j.EXTERNAL_JOB_ID,
            j.JOB_TEMPLATE_ID,
            j.JOB_TEMPLATES_EXTERNAL_ID,
            j.BLUESHIFT_REQUEST_ID,
            j.POSITION_ID,
            j.JOB_DESCRIPTION,
            j.JOB_ADDRESS_ID,
            j.JOB_SUPERVISOR_USER_ID,
            j.JOB_POSTER_ID,
            j.JOB_TYPE_ID,
            j.JOB_START_DATE_TIME,
            j.JOB_END_DATE_TIME,
            j.JOB_START_DATE_TIME_LOCAL,
            j.JOB_END_DATE_TIME_LOCAL,
            j.JOB_TIMEZONE,
            j.UPDATED_AT,
            j.JOB_NEEDED_LAST_COUNT,
            j.JOB_BATCH_SIZE,
            j.JOB_DAYS,
            j.JOB_SHIFTS,
            j.JOB_REASON_CODE,
            j.JOB_REASON_TEXT,
            j.JOB_ADDRESS,
            j.JOB_ADDRESS_LINE_TWO,
            j.JOB_ZIPCODE,
            j.JOB_ADDRESS_LATITUDE,
            j.JOB_ADDRESS_LONGITUDE
             */
        FROM tmp_first_shift fs
        LEFT JOIN BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE tsa
            ON tsa.JOB_ID = fs.JOB_ID
            AND tsa.START_TIME = fs.FIRST_SHIFT_START_TIME
            AND tsa.SEGMENT_INDEX = fs.FIRST_SEGMENT
            AND tsa.ACTIVE = TRUE
            --AND tsa._FIVETRAN_DELETED = FALSE -- account for duplicate rows bug
        LEFT JOIN BLUECREW.DM.DM_JOB_NEEDED_HISTORY nh
            ON nh.JOB_ID = fs.JOB_ID
            AND to_TIMESTAMP(START_TIME) BETWEEN START_DATE AND END_DATE
        INNER JOIN BLUECREW.DM.DM_JOBS j
            ON j.JOB_ID = fs.JOB_ID
--                 AND JOB_STATUS_ENUM < 6 -- active jobs only
        WHERE YEAR(to_TIMESTAMP(START_TIME)) >= 2020
            AND to_TIMESTAMP(START_TIME) <= DATEADD(DAY, 28, CURRENT_DATE())
            AND (j.INVITED_WORKER_COUNT IS NULL OR j.INVITED_WORKER_COUNT < COALESCE(nh.NEEDED, j.JOB_NEEDED_ORIGINAL_COUNT))
        ),
        tmp_sign_ups AS (
    -- Successful applications as of first shift start time
    SELECT
        fs.JOB_ID,
        jah.USER_ID,
        jah.APPLIED_STATUS_START_DATE AS SIGN_UP_TIME,
        DENSE_RANK() OVER (PARTITION BY jah.JOB_ID ORDER BY jah.APPLIED_STATUS_START_DATE ASC, USER_ID) AS SIGN_UP_ORDER
    FROM tmp_first_shift_full fs
    LEFT JOIN BLUECREW.DM.DM_CM_JOB_APPLIED_HISTORY jah
        ON fs.JOB_ID = jah.JOB_ID
        AND fs.START_TIME >= jah.APPLIED_STATUS_START_DATE
        AND fs.START_TIME < jah.END_DATE
        AND APPLIED_STATUS_ENUM = 0 -- successful sign up only
    ),

tmp_agg_sign_ups AS (
    -- Count of successful sign ups
    SELECT
        JOB_ID,
        COUNT(USER_ID) AS TOTAL_SUCCESSFUL_SIGN_UPS
    FROM tmp_sign_ups su
    GROUP BY JOB_ID),

tmp_first_shift_sign_up_count AS (
    -- Join sign-up counts
    SELECT
        fs.JOB_ID,
        fs.POSITION_ID,
        fs.SEGMENT_INDEX,
        fs.START_TIME,
        fs.START_TIME_LOCAL,
        fs.END_TIME,
        fs.END_TIME_LOCAL,
        fs.SCHEDULE_ID,
        fs.JOB_CREATED_AT,
        fs.JOB_TYPE,
        fs.JOB_TITLE,
    --             fs.JOB_DESCRIPTION,
        fs.JOB_NEEDED_ORIGINAL_COUNT,
        fs.JOB_WAGE,
        fs.JOB_STATUS_ENUM,
        fs.JOB_STATUS,
        fs.JOB_OVERFILL,
        fs.JOB_CITY,
        fs.JOB_STATE,
        fs.COMPANY_ID,
        fs.INVITED_WORKER_COUNT,
        fs.NEEDED,
        fs.JOB_SHIFTS,
        asu.TOTAL_SUCCESSFUL_SIGN_UPS,
        CASE
            -- no sign up so leave null
            WHEN asu.TOTAL_SUCCESSFUL_SIGN_UPS IS NULL THEN 0
            -- less than 100% fill, use max sign up count
            WHEN fs.NEEDED >= asu.TOTAL_SUCCESSFUL_SIGN_UPS THEN asu.TOTAL_SUCCESSFUL_SIGN_UPS
            -- more then 100% fill, use needed count
            WHEN fs.NEEDED < asu.TOTAL_SUCCESSFUL_SIGN_UPS THEN fs.NEEDED
            END AS SIGN_UP_JOIN_COUNT
    FROM tmp_first_shift_full fs
    LEFT JOIN tmp_agg_sign_ups asu
        ON fs.JOB_ID = asu.JOB_ID),

    tmp_successfully_worked as (
    SELECT job_id, segment_index, count(user_id) as successfully_worked
    FROM bluecrew.dm.dm_cm_hours_worked
    where (job_id, user_id) in (select job_id, user_id from tmp_sign_ups)
    GROUP BY 1, 2
    ),
    jacw as (
    select ssc.*, coalesce(sw.successfully_worked,0) as SUCCESSFULLY_WORKED
    from tmp_first_shift_sign_up_count ssc
    Left join tmp_successfully_worked sw
    on ssc.job_id = sw.job_id and ssc.segment_index = sw.segment_index
)

select jacw.*, c.COMPANY_ORIGIN, DATEDIFF(DAY, JOB_CREATED_AT, START_TIME) as POSTING_LEAD_TIME_DAYS
from jacw
left join bluecrew.dm.dm_companies c
on jacw.company_id = c.company_id
where 1=1
-- and job_type = 'Event Staff' 
and START_TIME >= '{start_date}'
and JOB_CREATED_AT >='{start_date}'
and START_TIME_LOCAL < '{end_date}'
""")



sdf = sdf.withColumn("JOB_ID",  sdf["JOB_ID"].cast('int'))
display(sdf)
print((sdf.count(), len(sdf.columns)))
# sdf

# COMMAND ----------

sdf = sdf.filter((sdf.NEEDED >=3)&(sdf.TOTAL_SUCCESSFUL_SIGN_UPS>=3)&(sdf.COMPANY_ORIGIN == 'BC')&(sdf.JOB_NEEDED_ORIGINAL_COUNT>=5)&(sdf.TOTAL_SUCCESSFUL_SIGN_UPS>=sdf.SUCCESSFULLY_WORKED)&(sdf.POSTING_LEAD_TIME_DAYS>0))
sdf = sdf.withColumn('Work', sdf.SUCCESSFULLY_WORKED / sdf.TOTAL_SUCCESSFUL_SIGN_UPS) 
display(sdf)


# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #This is a feature lookup that demonstrates how to use point in time based lookups for training sets
      FeatureLookup(
        table_name='feature_store.dev.jobs_data',
        feature_names=['COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'WAGE_DELTA', 'SCHEDULE_NAME_UPDATED',"ELIGIBLE_USERS", "ACTIVE_USERS_7_DAYS", "COUNTY", "ELIGIBLE_CMS_1_MILE", "ELIGIBLE_CMS_5_MILE","ELIGIBLE_CMS_10_MILE", 'ELIGIBLE_CMS_15_MILE', "ACTIVE_CMS_1_MILE", "ACTIVE_CMS_5_MILE", "ACTIVE_CMS_10_MILE", "ACTIVE_CMS_15_MILE", "JOB_TYPE_TITLE_COUNT", "TOTAL_JOB_COUNT", "TOTAL_CMS_REQUIRED", "CM_COUNT_RATIO"],
        lookup_key="JOB_ID",
        timestamp_lookup_key="JOB_CREATED_AT")
]
training_set = fe.create_training_set(
    df = sdf, # joining the original Dataset, with our FeatureLookupTable
    feature_lookups=model_feature_lookups,
    exclude_columns=[], # exclude id columns as we don't want them as feature
    label='Work'
)

training_pd = training_set.load_df()
display(training_pd)

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
cols_to_drop = ['SUCCESSFULLY_WORKED', 'TOTAL_SUCCESSFUL_SIGN_UPS', 'JOB_STATUS_ENUM', 'JOB_STATUS', 'JOB_OVERFILL', 'INVITED_WORKER_COUNT', 'SEGMENT_INDEX', 'NEEDED', 'SIGN_UP_JOIN_COUNT', 'POSITION_ID', 'COMPANY_ID', 'COUNTY_JOB_TYPE_TITLE_AVG_WAGE', 'SCHEDULE_ID']
df4 = df.drop(columns=cols_to_drop)
df5 = df4.set_index('JOB_ID')
df5

# COMMAND ----------

# Splits DF into train, test, and prediction dataframes
y = df5['Work']
X = df5.drop(columns=['Work'])

# Uses these for the training and testing
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# Checks for missing values and determines shape of X_train
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]

print("Columns with missing values :", cols_with_missing)
print("X_train_full shape :", X_train_full.shape)
print("X_valid_full shape :", X_valid_full.shape)

# COMMAND ----------

# IDs categorical and numeric columns for use in modeling and for splitting into proper pipeline.
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 30 and X_train_full[cname].dtype in ["object", "string"]]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int32', 'int64', 'float64', 'float32','decimal']]

print('categorical columns :', categorical_cols)
print('numerical columns :', numerical_cols)

# COMMAND ----------

# IDs new columns subset
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
print(my_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Pipeline

# COMMAND ----------

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
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

# COMMAND ----------

# DBTITLE 1,This is what we would use hyperopt to solve once we work out the shared cluster runtime issue
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from hyperopt import hp, fmin, tpe, STATUS_OK, SparkTrials
# from hyperopt.pyll.base import scope
# import shap
# import mlflow
# import numpy as np

# # Define your preprocessor here (as you did before)
# # preprocessor = ...

# # Include the model as a step in the pipeline
# my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                               ('model', RandomForestRegressor())])

# def train_model(params):
#     # Update the parameter names to reflect the pipeline structure
#     params = {'model__' + key: params[key] for key in params}
    
#     # Fit the pipeline with the training data
#     my_pipeline.set_params(**params).fit(X_train, y_train)

#     # SHAP values
#     # booster = my_pipeline.named_steps['model']
#     # shap_values = shap.TreeExplainer(booster).shap_values(X_train, y=y_train)
#     # shap.summary_plot(shap_values, X_train, feature_names=display_cols, plot_size=(14,6), max_display=10, show=False)
#     # plt.savefig("summary_plot.png", bbox_inches="tight") 
#     # plt.close()
#     # mlflow.log_artifact("summary_plot.png")

#     # Predict and evaluate
#     preds = my_pipeline.predict(X_valid)
#     (rmse, mae, r2) = eval_metrics(y_valid, preds)
#     # mlflow.log_metric("rmse", rmse)
#     # mlflow.log_metric("r2", r2)
#     # mlflow.log_metric("mae", mae)
#     return {'status': STATUS_OK, 'loss': rmse}

# # Define your search space
# search_space = {
#     'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
#     'n_estimators': scope.int(hp.quniform('n_estimators', 25, 150, 25))
# }

# # mlflow.autolog()

# # with mlflow.start_run():
# # Hyperopt optimization
# spark_trials = SparkTrials(parallelism=4)
# best_params = fmin(fn=train_model, space=search_space, algo=tpe.suggest, max_evals=50, trials=spark_trials, rstate=np.random.default_rng(seed=42))


# COMMAND ----------

# DBTITLE 1,This will be updated once the previous code runs
# Parameters from prior hyperparameter tuning
best_params = {'max_depth': 8.0, 'n_estimators': 150.0}

# COMMAND ----------

import mlflow

mlflow.autolog()


max_depth= int(best_params['max_depth'])
n_estimators=int(best_params['n_estimators'])
# Creates initial random forest model for training
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
with mlflow.start_run():
  # Preprocessing of training data, fit model 
  my_pipeline.fit(X_train, y_train)

  # Preprocessing of validation data, get predictions
  preds = my_pipeline.predict(X_valid)


  # Evaluate the model
  #Training Performance:
  print("Training Performance:")
  (rmse, mae, r2, mape) = eval_metrics(y_train, my_pipeline.predict(X_train))

  # Print out model metrics
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)
  print("  MAPE: %s" % mape)

  #Test Performance:
  print("Test Performance:")
  (rmse, mae, r2, mape) = eval_metrics(y_valid, preds)

  # Print out model metrics
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)
  print("  MAPE: %s" % mape)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)


# COMMAND ----------

# Initialize the MLflow client
client = mlflow.tracking.MlflowClient()

# Fetch all available experiments
experiments = client.search_experiments()

# Sort the experiments by their last update time in descending order
sorted_experiments = sorted(experiments, key=lambda x: x.last_update_time, reverse=True)

# Retrieve the most recently updated experiment
latest_experiment = sorted_experiments[0]

# Output the name of the latest experiment
print(f"The most recently updated experiment is named '{latest_experiment.name}'.")

# Note: If you're specifically looking for the experiment related to AutoML for base model creation,
# ensure that 'latest_experiment' corresponds to that experiment.

# COMMAND ----------

# Initialize the Databricks utilities to programmatically fetch the username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Retrieve the name of the latest experiment; assumed to have been set in earlier steps
experiment_name = latest_experiment.name

# Define the model name for the registry, specific to our use-case of Churn Prediction for a Bank
registry_model_name = "Overfill Test"

# Fetch the experiment details using its name
experiment_details = client.get_experiment_by_name(experiment_name)

# Search for runs within the experiment and sort them by validation F1 score in descending order
sorted_runs = mlflow.search_runs(experiment_details.experiment_id).sort_values("metrics.r2", ascending=False)

# Get the run ID of the best model based on the highest validation F1 score
best_run_id = sorted_runs.loc[0, "run_id"]

best_run_id
# Note: The variable `best_run_id` now contains the run ID of the best model in the specified experiment

# COMMAND ----------

# Initialize the model's URI using the best run ID obtained from previous steps
model_uri = f"runs:/{best_run_id}/model"

# Register the model in MLflow's model registry under the specified name
try:
    model_details = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
    print(f"Successfully registered model '{registry_model_name}' with URI '{model_uri}'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to register model '{registry_model_name}': {str(e)}")

model_details
# Note: The variable `model_details` now contains details about the registered model

# COMMAND ----------

# # Update the metadata of an already registered model
# try:
#     client.update_registered_model(
#         name=model_details.name,
#         description="This model predicts the show up rate for successful applications."
#     )
#     print(f"Successfully updated the description for the registered model '{model_details.name}'.")
# except mlflow.exceptions.MlflowException as e:
#     print(f"Failed to update the registered model '{model_details.name}': {str(e)}")

# # Update the metadata for a specific version of the model
# try:
#     client.update_model_version(
#         name=model_details.name,
#         version=model_details.version,
#         description="This is a scikit-learn random forest based model."
#     )
#     print(f"Successfully updated the description for version {model_details.version} of the model '{model_details.name}'.")
# except mlflow.exceptions.MlflowException as e:
#     print(f"Failed to update version {model_details.version} of the model '{model_details.name}': {str(e)}")

# # Note: The `model_details` variable is assumed to contain details about the registered model and its version

# COMMAND ----------

# Transition the model version to the 'Staging' stage in the model registry
try:
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True  # Archives any existing versions in the 'Staging' stage
    )
    print(f"Successfully transitioned version {model_details.version} of the model '{model_details.name}' to 'Staging'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to transition version {model_details.version} of the model '{model_details.name}' to 'Staging': {str(e)}")


# COMMAND ----------

# # the name of the model in the registry
# registry_model_name = "Overfill Test"

# # get the latest version of the model in staging and load it as a spark_udf.
# # MLflow easily produces a Spark user defined function (UDF).  This bridges the gap between Python environments and applying models at scale using Spark.
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{registry_model_name}/staging")

# COMMAND ----------

# # Predict on a Pandas DataFrame.
# import pandas as pd
# results = model.predict(X_valid)
# results

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation and Explanation

# COMMAND ----------

# Create a feature importance plot
feature_importance = FeatureImportance(my_pipeline)
feature_importance.plot(top_n_features=25)

# COMMAND ----------

# Builds DataFrame with predicted and actual values for test set
# Determines the range of values to consider for upper and lower bounds of a prediction interval.  Want this to be large enough to cover the actual work rate most of the time, but remain small enough to be useful
interval = .7
a = y_valid.reset_index()
a = a.rename(columns={'Work':'Actual_Show_Up_Rate'})
a['Predicted_Show_Up_Rate'] = preds
a['Delta'] = abs(a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate'])
a['Signed_Delta'] = a['Predicted_Show_Up_Rate'] - a['Actual_Show_Up_Rate']

# 50% interval
a['lowq'] = a['Signed_Delta'].quantile((1-interval)/2)
a['highq'] = a['Signed_Delta'].quantile(1-(1-interval)/2)

a['Dataset']="Test"
print(a)

# COMMAND ----------

# Builds DataFrame with predicted and actual values for training set
b = y_train.reset_index()
b = b.rename(columns={'Work':'Actual_Show_Up_Rate'})
b['Predicted_Show_Up_Rate'] = my_pipeline.predict(X_train)
b['Delta'] = abs(b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate'])
b['Signed_Delta'] = b['Predicted_Show_Up_Rate'] - b['Actual_Show_Up_Rate']
b['lowq'] = b['Signed_Delta'].quantile((1-interval)/2)
b['highq'] = b['Signed_Delta'].quantile(1-(1-interval)/2)
b['Dataset']="Training"
print(b)

# COMMAND ----------

# Merges job data to look at characteristics associated with predictions
c = pd.concat([a,b])

# creates the upper and lower bounds of the prediction interval
c['lower_guess']=(c['Predicted_Show_Up_Rate'] + c['lowq']).clip(lower = .01) #can't have negative numbers
c['upper_guess']=(c['Predicted_Show_Up_Rate'] + c['highq']).clip(upper = 1) #can't have numbers over 1

#merges with df5 to incorporate job data
eda_df = c.merge(df5, left_on='JOB_ID', right_index = True, how = 'left').sort_values('JOB_NEEDED_ORIGINAL_COUNT', ascending = True)
eda_df2 = eda_df[eda_df['Actual_Show_Up_Rate']==0]



# COMMAND ----------

overfill_added = eda_df.merge(df[['JOB_OVERFILL', 'TOTAL_SUCCESSFUL_SIGN_UPS', 'JOB_ID']], on='JOB_ID', how = 'left')
overfill_added['JOB_OVERFILL'].fillna(0,inplace=True)
overfill_added['PAST_PREDICTED_Show_Up_Rate']=overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']+overfill_added['JOB_OVERFILL'])
# original needed / original needed + overfill
overfill_added_test = overfill_added[overfill_added['Dataset']=='Test']
print("Past performance considering all jobs in test data:")

(rmse, mae, r2, mape) = eval_metrics(overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'Actual_Show_Up_Rate'], overfill_added_test.loc[overfill_added_test['START_TIME_LOCAL']<=end_date,'PAST_PREDICTED_Show_Up_Rate'])

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

overfill_added2 = overfill_added_test[overfill_added_test['JOB_OVERFILL']!=0]

print("Past performance on jobs with overfill added:")
(rmse, mae, r2, mape) = eval_metrics(overfill_added2['Actual_Show_Up_Rate'], overfill_added2['PAST_PREDICTED_Show_Up_Rate'])

# Print out model metrics
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)



# COMMAND ----------

#calculate costs associated with each prediction
overfill_added['Overfill_to_100']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Actual_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_to_100_perc'] = 1/overfill_added['Actual_Show_Up_Rate'] - 1
overfill_added['Overfill_Recommendation']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['Predicted_Show_Up_Rate']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_rec_perc'] = 1/overfill_added['Predicted_Show_Up_Rate'] - 1
overfill_added['Overfill_Rec_Upper_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['lower_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_upper_perc'] = 1/overfill_added['lower_guess'] - 1
overfill_added['Overfill_Rec_Lower_Bound']=(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']/overfill_added['upper_guess']-overfill_added['JOB_NEEDED_ORIGINAL_COUNT']).apply(np.round)
overfill_added['Overfill_lower_perc'] = 1/overfill_added['upper_guess'] - 1

overfill_added['Overfill_Rec_to_Ideal']=overfill_added['Overfill_Recommendation']-overfill_added['Overfill_to_100']
overfill_added['Overfill_Past_to_Ideal']=overfill_added['JOB_OVERFILL']-overfill_added['Overfill_to_100']

overfill_added['Overfill_Prediction_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']*-25 if row['Overfill_Rec_to_Ideal']<0 else 0, axis=1).apply(np.round)
overfill_added['Overfill_Prediction_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else 0, axis=1).apply(np.round)

overfill_added['Overfill_Past_Rec_Lost_Profit'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']*-25 if row['Overfill_Past_to_Ideal']<0 else 0, axis=1).apply(np.round)
overfill_added['Overfill_Past_Rec_CM_Payout'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else 0, axis=1).apply(np.round)

overfill_added['Overfill_Prediction_Cost'] = overfill_added.apply(lambda row: row['Overfill_Rec_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Rec_to_Ideal']>=0 else row['Overfill_Rec_to_Ideal']*-25, axis=1).apply(np.round)
overfill_added['Overfill_Past_Rec_Cost'] = overfill_added.apply(lambda row: row['Overfill_Past_to_Ideal']* row['JOB_WAGE']*4 if row['Overfill_Past_to_Ideal']>=0 else row['Overfill_Past_to_Ideal']*-25, axis=1).apply(np.round)
overfill_added[overfill_added['Actual_Show_Up_Rate']>0]
overfill_added[(overfill_added['Dataset']=='Future')&(overfill_added['JOB_ID']==329636)]

# COMMAND ----------



# COMMAND ----------

cost_estimate = overfill_added[overfill_added['Actual_Show_Up_Rate']>0]
print(cost_estimate.shape)
print('Model Overfill Prediction_Cost:' + str(cost_estimate['Overfill_Prediction_Cost'].sum()))
print('Past Overfill Prediction_Cost:' + str(cost_estimate['Overfill_Past_Rec_Cost'].sum()))
print('Model Overfill Lost Profit:' + str(cost_estimate['Overfill_Prediction_Lost_Profit'].sum()))
print('Past Overfill Lost Profit:' + str(cost_estimate['Overfill_Past_Rec_Lost_Profit'].sum()))
print('Model Overfill CM Payout:' + str(cost_estimate['Overfill_Prediction_CM_Payout'].sum()))
print('Past Overfill CM Payout:' + str(cost_estimate['Overfill_Past_Rec_CM_Payout'].sum()))

# COMMAND ----------

x_array = np.arange(0,1,.01)
x_array2 = np.arange(0,3500,1)

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_Cost",
                    y="Overfill_Past_Rec_Cost",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[(overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50)&(overfill_added['Dataset']!="Future")])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill Costs for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_Lost_Profit",
                    y="Overfill_Past_Rec_Lost_Profit",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill Lost Profit for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

#Used to see if job size impacted model accuracy.  Used this to determine jobs with 1 or 2 people were throwing off the model.
sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Overfill_Prediction_CM_Payout",
                    y="Overfill_Past_Rec_CM_Payout",
                    hue = 'JOB_NEEDED_ORIGINAL_COUNT',
                    data=overfill_added[overfill_added['JOB_NEEDED_ORIGINAL_COUNT']<=50])
plt.axis('equal')
# plt.xlim(0,14000)
plt.title("Overfill CM Payouts for Jobs with Less Than 50 Requested")
plt.plot(x_array2, x_array2, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Actual_Show_Up_Rate",
                    y="Predicted_Show_Up_Rate",
                    hue = 'Delta',
                    data=overfill_added_test)
plt.plot(x_array, x_array, color='black')
xlim=np.array(plt.gca().get_xlim())
ylim=np.array(plt.gca().get_ylim())
plt.fill_between(xlim, y1=xlim, y2=[ylim[0],ylim[0]], 
                 color="#e0eaf3", zorder=0 )
plt.fill_between(xlim, y1=xlim, y2=[ylim[1],ylim[1]], 
                 color="#fae4e4", zorder=0)
plt.margins(0)
plt.xlabel('Actual Show Up Rate')
plt.ylabel('Predicted Show Up Rate')
plt.show()

# COMMAND ----------

sns.set(style='whitegrid')
plt.figure(figsize=(10,10))
sns.scatterplot(x="Actual_Show_Up_Rate",
                    y="PAST_PREDICTED_Show_Up_Rate",
                    hue = 'Dataset',
                    data=overfill_added2)
plt.show()

# COMMAND ----------

df = spark.createDataFrame(overfill_added)
df.createOrReplaceTempView('data')

# COMMAND ----------

df.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL').save()


# COMMAND ----------

display(df[df['Actual_Show_Up_Rate']==1])

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


