# Databricks notebook source
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

# COMMAND ----------

from pyspark.sql import DataFrame
def jobs_query(start_date: str, end_date: str) -> DataFrame:
    """
    Runs SQL query to pull job-based show up rate data
    
    Parameters:
    - start_date: The first date of a job created to consider in the query.
    - end_date: The last date of a job start time local to consider in the query.  Usually, this date needs to be in the past to ensure work data is accurate in the dataset.
    
    Returns:
    - Spark DataFrame: The resulting Spark DataFrame.
    """
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
    return sdf
