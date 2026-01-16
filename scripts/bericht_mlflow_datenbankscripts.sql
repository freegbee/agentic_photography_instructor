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
    m_eval.run_uuid = '105c494bca5943cdbb78d7365291d505'
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
    m_eval.run_uuid = '105c494bca5943cdbb78d7365291d505'
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
    REPLACE(k.key, 'train_transformer_usage/', '') AS layer_metric,
    (s.step / 4000) as step,
    m.value
FROM
    -- 1. Alle Steps generieren (X-Achse Gerüst)
    generate_series(4000, 300000, 4000) AS s(step)
CROSS JOIN
    -- 2. Alle vorkommenden Keys ermitteln (Y-Achse Gerüst)
    (SELECT DISTINCT key FROM metrics
     WHERE run_uuid = 'eefeb7e047784d5e9584276505d2f12c' AND key LIKE 'train_transformer_usage/%') k
LEFT JOIN
    metrics m
    ON m.step = s.step
    AND m.key = k.key
    AND m.run_uuid = 'eefeb7e047784d5e9584276505d2f12c'
ORDER BY
    layer_metric, s.step;