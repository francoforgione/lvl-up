CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES app.conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    model TEXT,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app.retrieved_chunks (
    id BIGSERIAL PRIMARY KEY,
    message_id UUID NOT NULL REFERENCES app.messages(id),
    chunk_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS app.feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id UUID NOT NULL REFERENCES app.messages(id),
    rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app.eval_runs (
    id BIGSERIAL PRIMARY KEY,
    run_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
