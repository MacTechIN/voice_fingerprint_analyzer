-- VoiceGuard Verification — Phase 2 스키마
--
-- Supabase(PostgreSQL + pgvector)에 그대로 적용 가능하다.
-- 적용: psql "$DATABASE_URL" -f migrations/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 화자 등록 성문
-- ---------------------------------------------------------------------------
-- 벡터와 함께 model/dim을 반드시 저장한다. 모델을 교체하면 기존 벡터와
-- 호환되지 않아 재등록이 필요한데, 메타데이터가 없으면 어떤 행이 어느 모델
-- 것인지 사후에 알 수 없다 (02 §6 재등록 마이그레이션).
--
-- 차원을 192로 고정한 이유: pgvector의 vector(N)은 고정 차원을 요구하고,
-- HNSW 인덱스도 차원별로 만들어진다. Phase 7에서 백본을 바꿔 차원이 달라지면
-- 새 테이블(또는 새 컬럼)로 마이그레이션한다 — 같은 컬럼에 섞어 담을 수 없다.
CREATE TABLE IF NOT EXISTS speaker_enrollments (
    id              BIGSERIAL    PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    embedding       vector(192) NOT NULL,
    model           TEXT        NOT NULL,
    dim             INTEGER     NOT NULL,
    l2_normalized   BOOLEAN     NOT NULL DEFAULT TRUE,
    speech_duration_sec REAL,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT speaker_enrollments_dim_matches CHECK (dim = 192)
);

CREATE INDEX IF NOT EXISTS speaker_enrollments_user_active_idx
    ON speaker_enrollments (user_id) WHERE is_active;

-- 1:1 검증(FR-05)은 user_id로 좁힌 뒤 소수의 벡터만 비교하므로 ANN 인덱스가
-- 필요 없다. HNSW는 1:N 식별(향후 확장)을 도입할 때 추가한다 — 그때는
-- vector_cosine_ops로 만든다.

-- ---------------------------------------------------------------------------
-- 검증 시도 감사 로그
-- ---------------------------------------------------------------------------
-- 03 문서의 오딧 트레일·품질 지표 대시보드가 이 테이블을 읽는다.
-- 원시 점수와 (Phase 6 이후) 정규화 점수를 함께 남겨 정규화 효과를 추적한다.
CREATE TABLE IF NOT EXISTS verification_attempts (
    id              BIGSERIAL    PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    raw_cosine      REAL,
    normalized_score REAL,      -- Phase 6 AS-Norm. 그전까지는 NULL.
    match_probability REAL,
    is_verified     BOOLEAN,
    threshold       REAL,
    model           TEXT,
    speech_duration_sec REAL,
    outcome         TEXT        NOT NULL,  -- verified | rejected | not_enrolled | audio_rejected
    error_code      TEXT,
    client_ip       TEXT,
    elapsed_ms      REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS verification_attempts_user_time_idx
    ON verification_attempts (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS verification_attempts_time_idx
    ON verification_attempts (created_at DESC);

-- ---------------------------------------------------------------------------
-- (Phase 6 예정) AS-Norm 임포스터 코호트
-- ---------------------------------------------------------------------------
-- 사용자 성문과 물리적으로 분리한다. 코호트는 실제 사용자와 무관한 화자의
-- 임베딩이며, 사용자 테이블과 섞이면 사칭자 통계가 오염된다 (02 §4.2).
-- 테이블은 Phase 6에서 생성한다.
