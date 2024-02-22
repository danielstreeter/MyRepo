with distance as (
  select 3959 * 2 * ASIN(SQRT(POWER(SIN((user_address_latitude
          - job_address_latitude) * PI() / 180 / 2), 2) + COS(user_address_latitude * PI() / 180) * COS(job_address_latitude * PI() / 180) * POWER(SIN((user_address_longitude - job_address_longitude) * PI() / 180 / 2), 2))) AS distance_miles,
          *
  from dm.dm_users u
  join dm.dm_jobs j 
  where 1 = 1
  and j.job_id = 329730
),
distance_eligible_users as (
  select * 
  from distance d
  where distance_miles <=70
  and user_status_enum = 1
),
successful_applications as (
  select jas.* from dm.dm_cm_job_application_status jas
  left join dm.dm_jobs j
  on jas.job_id = j.job_id
  where user_id in (select user_id from distance_eligible_users)
  and USER_CURRENT_JOB_APPLICATION_STATUS in ('SUCCESS', 'ADDED_TO_WAITLIST', 'PENDING_WAITLIST_CHECK')
  and j.job_status_enum = 1
  and j.job_end_date_time >= sysdate()
),
applied_shifts as (
  select user_id, sa.job_id, segment_index, to_timestamp_ntz(start_time) as start_time, to_timestamp_ntz(end_time) as end_time 
  from successful_applications sa
  left join BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE tsa
  on sa.job_id = tsa.job_id
  where to_timestamp_ntz(end_time)>=sysdate()
  and active = 1
),
job_shifts as (
  select job_id, segment_index, to_timestamp_ntz(start_time) as start_time, to_timestamp_ntz(end_time) as end_time 
  from BLUECREW.MYSQL_BLUECREW.TIME_SEGMENTS_ABSOLUTE
  where job_id in (select job_id from distance)
  and to_timestamp_ntz(end_time)>=sysdate()
  and active = 1
),
ineligible_users as (
  select distinct user_id
  from applied_shifts a
  inner join job_shifts js 
  on 
      js.start_time between dateadd(HOUR,-6, a.start_time) and dateadd(HOUR,6, a.end_time)
      or js.end_time between dateadd(HOUR,-6, a.start_time) and dateadd(HOUR,6, a.end_time)

)

select u.* 
from distance_eligible_users deu
left join dm.fact_users u
on deu.user_id = u.user_id
where deu.user_id not in (select * from ineligible_users)