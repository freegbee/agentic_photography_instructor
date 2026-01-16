select * from experiments e 
-- experiment 57

select *
from   runs r 
where  r.experiment_id = 57
-- run-id eefeb7e047784d5e9584276505d2f12c

select *
from   logged_model_metrics lmm 
where  lmm.run_id = 'eefeb7e047784d5e9584276505d2f12c'

select step, value
from   metrics m
where  key='train/success_rate'
left join 
(select step, value
from   metrics m 
where  m.run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
and    key='eval/success_rate'
order by step asc) eval_success

select step, value
from   metrics m
where  m.run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
and    key='train/success_rate'


-- ========= success rate
SELECT
    m_eval.step AS eval_step,
    m_eval.value AS eval_success_rate,
    -- m_train.step AS train_step,
    m_train.value AS train_success_rate
FROM metrics m_eval
JOIN metrics m_train
    ON m_eval.run_uuid = m_train.run_uuid
    -- Hier ist die geforderte Logik: Train Step = 4000 * Eval Step
    AND m_train.step = m_eval.step
WHERE
    m_eval.run_uuid = '0a0deea790034af3bb521bbc61dbbcbe'
    AND m_eval.key = 'eval/success_rate'
    AND m_train.key = 'train/success_rate'
ORDER BY
    m_eval.step ASC;


-- ========= mean reward

SELECT
    m_eval.step AS eval_step,
    m_eval.value AS eval_mean_reward,
    -- m_train.step AS train_step,
    m_train.value AS train_mean_reward
FROM metrics m_eval
JOIN metrics m_train
    ON m_eval.run_uuid = m_train.run_uuid
    -- Hier ist die geforderte Logik: Train Step = 4000 * Eval Step
    AND m_train.step = m_eval.step
WHERE
    m_eval.run_uuid = '0a0deea790034af3bb521bbc61dbbcbe'
    AND m_eval.key = 'eval/mean_reward'
    AND m_train.key = 'train/mean_reward'
ORDER BY
    m_eval.step ASC;

-- ========= STOP Actions
SELECT
    m_eval.step AS timestep,
    m_eval.value AS eval_transformer_usage_STOP,
    -- m_train.step AS train_step,
    m_train.value AS train_stransformer_usage_STOP
FROM metrics m_eval
JOIN metrics m_train
    ON m_eval.run_uuid = m_train.run_uuid
    -- Hier ist die geforderte Logik: Train Step = 4000 * Eval Step
    AND m_train.step = m_eval.step
WHERE
    m_eval.run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
    AND m_eval.key = 'eval_transformer_usage/STOP'
    AND m_train.key = 'train_transformer_usage/STOP'
ORDER BY
    m_eval.step ASC;


-- Vergleich der drei haupttrainings:
-- eval/mean_reward
-- eval/mean_episodes_len
-- eval_transformer_usage/STOP
-- eval_transformer_usage/CA_WARMTH
SELECT
    s.step,
    MAX(m.value) FILTER (WHERE m.run_uuid = 'eefeb7e047784d5e9584276505d2f12c') AS "Haupttraining 1",
    MAX(m.value) FILTER (WHERE m.run_uuid = '105c494bca5943cdbb78d7365291d505') AS "Haupttraining 2",
    MAX(m.value) FILTER (WHERE m.run_uuid = '0a0deea790034af3bb521bbc61dbbcbe') AS "Haupttraining 3"
FROM
    -- Generiert Steps 0 bis 74
    generate_series(0, 74, 1) AS s(step)
LEFT JOIN
    metrics m
    ON m.key = 'eval_transformer_usage/CA_COOLNESS'
    AND m.run_uuid IN ('eefeb7e047784d5e9584276505d2f12c', '105c494bca5943cdbb78d7365291d505', '0a0deea790034af3bb521bbc61dbbcbe')
    -- HIER ist die korrigierte Logik:
    -- Wir schauen, ob der normalisierte Step aus der DB (1, 2, 3...) 
    -- zum generierten Step + 1 passt.
    AND (s.step + 1) = (
        CASE 
            WHEN m.run_uuid = '0a0deea790034af3bb521bbc61dbbcbe' THEN m.step / 4032
            ELSE m.step / 4000
        END
    )
GROUP BY
    s.step
ORDER BY
    s.step ASC;


SELECT
    s.step,
    MAX(value) FILTER (WHERE run_uuid = 'eefeb7e047784d5e9584276505d2f12c') AS "Haupttraining 1",
    MAX(value) FILTER (WHERE run_uuid = '105c494bca5943cdbb78d7365291d505') AS "Haupttraining 2",
    MAX(value) FILTER (WHERE run_uuid = '0a0deea790034af3bb521bbc61dbbcbe') AS "Haupttraining 3"
FROM
    -- 1. Synthetische Steps generieren (Start, Ende, Schrittweite)
    generate_series(0, 74, 1) AS s(step)
LEFT JOIN
    metrics m
    ON ((s.step + 0) * 1) = m.step
    AND key = 'eval/mean_episodes_len'
    AND run_uuid IN ('eefeb7e047784d5e9584276505d2f12c', '105c494bca5943cdbb78d7365291d505', '0a0deea790034af3bb521bbc61dbbcbe')
GROUP BY
    s.step
ORDER BY
    s.step ASC;



select *
from   metrics m 
where  1=1
and    run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
and    key = 'eval/mean_episodes_len'

-- =============== HEatmap
SELECT
    REPLACE(k.key, 'eval_transformer_usage/', '') AS layer_metric,
    (s.step / 4000) as step,
    m.value
FROM
    -- 1. Alle Steps generieren (X-Achse Gerüst)
    generate_series(4000, 300000, 4000) AS s(step)
CROSS JOIN
    -- 2. Alle vorkommenden Keys ermitteln (Y-Achse Gerüst)
    (SELECT DISTINCT key FROM metrics
     WHERE run_uuid = 'eefeb7e047784d5e9584276505d2f12c' AND key LIKE 'eval_transformer_usage/%') k
LEFT JOIN
    metrics m
    ON m.step = s.step
    AND m.key = k.key
    AND m.run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
ORDER BY
    layer_metric, s.step;


-- =================== Überblick finalized runs
WITH target_runs AS (
    -- 1. Schritt: Nur die Run-UUIDs holen, die den gewünschten Tag haben
    SELECT run_uuid
    FROM tags
    WHERE key = 'finalized'       -- z.B. 'experiment_group'
      AND value = 'true'    -- z.B. 'baseline_v2'
),
run_params AS (
    -- 2. Schritt: Parameter pivotieren (Zeilen zu Spalten)
    -- Hier definieren Sie, welche Parameter Sie sehen wollen
    -- task_params/success_bonus, task_params/success_bonus_strategy, task_params/step_penalty, task_params/reward_strategy
    SELECT
        p.run_uuid,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/ppo_model_variant') AS model_variant,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/model_learning_schedule') AS learning_schedule,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/gamma') AS gamma,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/clip_range') AS clip_range,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/ent_coef') AS ent_coef,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/net_arch') AS net_arch,
        MAX(p.value) FILTER (WHERE p.key = 'task_params/success_bonus_strategy') AS success_bonus_strategy,
        MAX(p.value) FILTER (WHERE p.key = 'task_params/success_bonus') AS success_bonus,
        MAX(p.value) FILTER (WHERE p.key = 'task_params/reward_strategy') AS reward_strategy,
        MAX(p.value) FILTER (WHERE p.key = 'task_params/step_penalty') AS step_penalty,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/batch_size') AS batch_size,
        MAX(p.value) FILTER (WHERE p.key = 'ppo_model_params/rollout_size') AS rollout_size
    FROM params p
    JOIN target_runs t ON p.run_uuid = t.run_uuid
    GROUP BY p.run_uuid
),
run_stats AS (
    -- 3. Schritt: Metriken aggregieren (Mean, Q1, Q3)
    SELECT
        m.run_uuid,
        -- --- Eval Metrics ---
        AVG(m.value) FILTER (WHERE key = 'eval/mean_reward') AS eval_mean,
        -- 1. Quartil (25%)
        percentile_cont(0.25) WITHIN GROUP (ORDER BY m.value) 
            FILTER (WHERE key = 'eval/mean_reward') AS eval_q1,
        -- 3. Quartil (75%)
        percentile_cont(0.75) WITHIN GROUP (ORDER BY m.value) 
            FILTER (WHERE key = 'eval/mean_reward') AS eval_q3,
        -- --- Train Metrics ---
        AVG(m.value) FILTER (WHERE key = 'train/mean_reward') AS train_mean,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY m.value) 
            FILTER (WHERE key = 'train/mean_reward') AS train_q1,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY m.value) 
            FILTER (WHERE key = 'train/mean_reward') AS train_q3
    FROM metrics m
    JOIN target_runs t ON m.run_uuid = t.run_uuid
    WHERE m.key IN ('eval/mean_reward', 'train/mean_reward')
    GROUP BY m.run_uuid
),
run_description as (
   select
       r.run_uuid,
       r.lifecycle_stage, 
       r.end_time,
       to_timestamp(r.end_time/ 1000.0) AT TIME ZONE 'Europe/Zurich' AS lokal_zeit_komplett,
       to_char(to_timestamp(r.end_time/ 1000.0) AT TIME ZONE 'Europe/Zurich', 'DD.MM.YYYY HH24:MI:SS') AS formatierter_string
   from runs r
   JOIN target_runs t ON r.run_uuid = t.run_uuid
   GROUP BY r.run_uuid
)
-- 4. Schritt: Alles zusammenfügen
SELECT
    t.run_uuid,
    -- Kopfwerte
    r.formatierter_string as ende,
    -- Parameter Spalten
    p. model_variant,
    p.learning_schedule,
    p.gamma,
    p.clip_range,
    p.ent_coef,
    p.net_arch,
    p.success_bonus_strategy,
    p.success_bonus ,
    p.reward_strategy,
    p.step_penalty ,
    p.batch_size,
    -- Eval Statistik
    s.eval_mean,
    s.eval_q1,
    s.eval_q3,
    -- Train Statistik
    s.train_mean,
    s.train_q1,
    s.train_q3
FROM target_runs t
LEFT JOIN run_params p ON t.run_uuid = p.run_uuid
LEFT JOIN run_stats s ON t.run_uuid = s.run_uuid
left join run_description r on t.run_uuid = r.run_uuid
where r.lifecycle_stage = 'active'
order by r.end_time;


select * from runs;

select *        
   from runs r