-- Example warehouse query: weighted user-app evidence before the evaluation cutoff.
-- Replace :train_cutoff with a fixed timestamp in a production backtest.
SELECT
    user_id,
    app_id,
    SUM(CASE event_name
        WHEN 'af_view' THEN 0.2
        WHEN 'af_click' THEN 1.0
        WHEN 'af_open' THEN 1.0
        WHEN 'af_install' THEN 3.0
        WHEN 'af_purchase' THEN 4.0
        WHEN 'af_complete_registration' THEN 3.0
        WHEN 'af_level_achieved' THEN 2.0
        ELSE 1.0
    END) AS weighted_event_count,
    MAX(event_time) AS last_event_time,
    COUNT(*) AS event_count
FROM interaction_events
WHERE event_time < :train_cutoff
  AND user_id IS NOT NULL
  AND app_id IS NOT NULL
GROUP BY user_id, app_id;


-- Example app eligibility join; the event timestamp cutoff must be applied
-- before joining time-varying metadata to avoid future-information leakage.
SELECT i.user_id, i.app_id, i.event_time, i.event_name
FROM interaction_events AS i
JOIN app_dimension AS a ON i.app_id = a.app_id
WHERE i.event_time < :train_cutoff
  AND a.status = 'active';
