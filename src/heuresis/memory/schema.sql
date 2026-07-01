-- Memory primitive schema. One DB per campaign (exp.dir/memory.db).
-- Idempotent: re-running these statements on an existing DB is a no-op.

-- Framework-written. One row per executor completion. The idea is the
-- anchor; notes.md (if produced by the executor) gets folded into the
-- embedded text.
CREATE TABLE IF NOT EXISTS experiments (
  ideator_id      TEXT NOT NULL,           -- workspace UUID of the ideator that proposed this
  executor_id     TEXT NOT NULL,           -- workspace UUID of the executor that ran it
  ts              REAL NOT NULL,
  valid           INTEGER NOT NULL,        -- 0/1
  score           REAL,                    -- nullable if invalid
  features_json   TEXT,                    -- {"axis_a": 0.42, ...} (optional per algo)
  parent_ids_json TEXT,                    -- ["<exec-uuid>", ...]
  generation      INTEGER,
  idea_md         TEXT NOT NULL,           -- the idea that started this experiment
  PRIMARY KEY (ideator_id, executor_id)
);

-- Agent-written insights. Append-only in v1.
CREATE TABLE IF NOT EXISTS learnings (
  learning_id               INTEGER PRIMARY KEY AUTOINCREMENT,
  ts                        REAL NOT NULL,
  author_id                 TEXT NOT NULL,   -- workspace UUID
  author_role               TEXT NOT NULL,   -- "ideator" | "executor"
  tags_json                 TEXT,            -- ["cma-es", "rastrigin", ...]
  related_executor_ids_json TEXT,            -- optional backrefs: ["<exec-uuid>", ...]
  content                   TEXT NOT NULL
);

-- Vector indexes (sqlite-vec). vec0 virtual tables are separate from the
-- base tables, so inserts into experiments/learnings don't auto-populate
-- these. MemoryStore writes to both in the same transaction.
--
-- Schema note: vec0 supports one PK column (not composite). ``executor_id``
-- is a workspace UUID, unique per execution, so it's sufficient as PK.
-- ``ideator_id`` rides along as an auxiliary column (the ``+`` prefix
-- marks it as metadata — not indexed, readable after a match).
CREATE VIRTUAL TABLE IF NOT EXISTS experiments_vec USING vec0(
  executor_id TEXT PRIMARY KEY,
  +ideator_id TEXT,
  embedding   float[3072]
);

CREATE VIRTUAL TABLE IF NOT EXISTS learnings_vec USING vec0(
  learning_id INTEGER PRIMARY KEY,
  embedding   float[3072]
);

-- Read-only views are the only surface exposed to agents via `memory read`.
-- SQLite views are read-only by default (no INSTEAD OF triggers defined).
-- Defined as 1:1 projections so we can refactor underlying tables later
-- without breaking agent SQL.
CREATE VIEW IF NOT EXISTS memory_experiments_v AS
  SELECT ideator_id, executor_id, ts, valid, score,
         features_json, parent_ids_json, generation, idea_md
  FROM experiments;

CREATE VIEW IF NOT EXISTS memory_learnings_v AS
  SELECT learning_id, ts, author_id, author_role,
         tags_json, related_executor_ids_json, content
  FROM learnings;
