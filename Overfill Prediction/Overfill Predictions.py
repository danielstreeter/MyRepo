# Databricks notebook source
# MAGIC %md
# MAGIC # Job Level Model
# MAGIC This notebook loads snowflake data, cleans it, and develops a model to predict job-level show up rates (worked first shift/successful applications at job start) that will help with setting an overfill rate.  This model is at the job level and doesn't consider the attricutes of CMs that have applied.  Since it only considers successful applications at the time of job start, all cancellations prior to job start are excluded.

# COMMAND ----------

# MAGIC %md
# MAGIC #Preprocessing

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
from pyspark.sql.functions import to_date, current_timestamp
import pyspark.sql.functions as F
import mlflow
from mlflow.models.signature import infer_signature


# COMMAND ----------

# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    return rmse, mae, r2, mape

# COMMAND ----------

# Setting up database access
sfUser = dbutils.secrets.get(scope="my_secrets", key="snowflake-user")
SFPassword = dbutils.secrets.get(scope="my_secrets", key="snowflake-password")
 
options = {
  "sfUrl": "vha09841.snowflakecomputing.com",
  "sfUser": sfUser,
  "SFPassword": SFPassword,
  "sfDataBase": "BLUECREW",
  "sfSchema": "PERSONALIZATION",
  "sfWarehouse": "COMPUTE_WH"
}

now = datetime.now()
start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

# COMMAND ----------

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
and START_TIME_LOCAL >= '{start_date}'
""")



sdf = sdf.withColumn("JOB_ID",  sdf["JOB_ID"].cast('int')).withColumn("START_TIME",to_date("START_TIME"))

# sdf

# COMMAND ----------

sdf = sdf.filter((sdf.NEEDED >0)).withColumn('Work', sdf.SUCCESSFULLY_WORKED / sdf.TOTAL_SUCCESSFUL_SIGN_UPS) 
display(sdf)


# COMMAND ----------

# This looks up job-level features in feature store
# # For more info look here: https://docs.gcp.databricks.com/en/machine-learning/feature-store/time-series.html
fe = FeatureEngineeringClient()
model_feature_lookups = [
      #This is a basic feature lookup that doesn't have a timestamp key
    #   FeatureLookup(
    #       table_name='feature_store.dev.calendar',
    #       lookup_key="START_TIME"
    #   )
    #   ,
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
# display(training_pd)

# COMMAND ----------

# Converts the Spark df + features to a pandas DataFrame and turns boolean columns into integers
# convert this to do the type casting first and then use toPandas()?
for idx, (col, dtype) in enumerate(zip(training_pd.columns, training_pd.dtypes)):
  if dtype[1].startswith('decimal'):
    training_pd = training_pd.withColumn(col, F.col(col).cast('float'))

df = training_pd.toPandas()
# df = df.astype({'JOB_WAGE':'float', 'COUNTY_JOB_TYPE_TITLE_AVG_WAGE':'float', 'WAGE_DELTA':'float', 'JOB_OVERFILL':'float', 'JOB_SHIFTS':'float', 'INVITED_WORKER_COUNT':'float','POSTING_LEAD_TIME_DAYS':'float',"ELIGIBLE_USERS":'float', "ACTIVE_USERS_7_DAYS":'float', 'SUCCESSFULLY_WORKED':'float', 'TOTAL_SUCCESSFUL_SIGN_UPS':'float', "ELIGIBLE_CMS_1_MILE":'float', "ELIGIBLE_CMS_5_MILE":'float',"ELIGIBLE_CMS_10_MILE":'float', 'ELIGIBLE_CMS_15_MILE':'float', "ACTIVE_CMS_1_MILE":'float', "ACTIVE_CMS_5_MILE":'float', "ACTIVE_CMS_10_MILE":'float', "ACTIVE_CMS_15_MILE":'float', "JOB_TYPE_TITLE_COUNT":'float', "TOTAL_JOB_COUNT":'float', "TOTAL_CMS_REQUIRED":'float', 'JOB_NEEDED_ORIGINAL_COUNT':'float'})

bool_cols = [cname for cname in df.columns if df[cname].dtype == 'bool']
for col in bool_cols:
  df[col] = df[col].astype(int)
print(list(df.columns))
df.set_index('JOB_ID', inplace = True)

# COMMAND ----------

input_cols = ['JOB_TYPE', 'JOB_STATE', 'COMPANY_ORIGIN', 'SCHEDULE_NAME_UPDATED', 'JOB_NEEDED_ORIGINAL_COUNT', 'JOB_WAGE', 'JOB_SHIFTS', 'POSTING_LEAD_TIME_DAYS', 'WAGE_DELTA', 'ELIGIBLE_USERS', 'ACTIVE_USERS_7_DAYS', 'ELIGIBLE_CMS_1_MILE', 'ELIGIBLE_CMS_5_MILE', 'ELIGIBLE_CMS_10_MILE', 'ELIGIBLE_CMS_15_MILE', 'ACTIVE_CMS_1_MILE', 'ACTIVE_CMS_5_MILE', 'ACTIVE_CMS_10_MILE', 'ACTIVE_CMS_15_MILE', 'JOB_TYPE_TITLE_COUNT', 'TOTAL_JOB_COUNT', 'TOTAL_CMS_REQUIRED', 'CM_COUNT_RATIO']
df2 = df[input_cols]
df2

# COMMAND ----------

# the name of the model in the registry
registry_model_name = "Overfill Test"

# get the latest version of the model in staging and load it as a spark_udf.
# MLflow easily produces a Spark user defined function (UDF).  This bridges the gap between Python environments and applying models at scale using Spark.
model = mlflow.pyfunc.load_model(model_uri=f"models:/{registry_model_name}/staging")

# COMMAND ----------

df2['Work'] = model.predict(df2)

# COMMAND ----------


df2

# COMMAND ----------

# MAGIC %sql
# MAGIC -- select TOTAL_CMS_REQUIRED/TOTAL_JOB_COUNT as ratio, * from feature_store.dev.jobs_data2
# MAGIC -- where job_id in (322963, 322957, 322962)
# MAGIC select * from feature_store.dev.jobs_data
# MAGIC where CM_COUNT_RATIO is null

# COMMAND ----------

# MAGIC %md
# MAGIC # Need to add the post processing outputs to get to new predictions for Streamlit

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

# Builds DataFrame with predicted and actual values for future set.  Don't have actual values for these.
d = y_future.reset_index()
d = d.rename(columns={'Work':'Actual_Show_Up_Rate'})

d['Predicted_Show_Up_Rate'] = my_pipeline.predict(future_jobs)
d['Actual_Show_Up_Rate'] = 0
d['Delta'] = 0
d['Signed_Delta'] = 0
#signed delta from test dataset
d['lowq'] = a['Signed_Delta'].quantile((1-interval)/2)
d['highq'] = a['Signed_Delta'].quantile(1-(1-interval)/2)
d['Dataset']="Future"

d

# COMMAND ----------

# Merges job data to look at characteristics associated with predictions
c = pd.concat([a,b,d])

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

df.write.format("snowflake").options(**options).mode("overwrite").option("dbtable", 'OVERFILL_PREDICTIONS').save()


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


