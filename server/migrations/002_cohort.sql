-- Phase 6: AS-Norm 임포스터 코호트
--
-- 사용자 성문(speaker_enrollments)과 **물리적으로 분리한다**. 코호트는 실제
-- 사용자와 무관한 화자의 임베딩이며, 사용자 테이블과 섞이면 정규화가 자기
-- 자신을 참조하게 되어 사칭자 통계가 오염된다 (02 §4.2).

CREATE TABLE IF NOT EXISTS impostor_cohort (
    id            BIGSERIAL   PRIMARY KEY,
    embedding     vector(192) NOT NULL,
    model         TEXT        NOT NULL,
    dim           INTEGER     NOT NULL,
    source        TEXT,       -- 출처 데이터셋 (예: librispeech/test-clean)
    speaker_ref   TEXT,       -- 원 화자 식별자. 동일 화자 편중 점검용
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT impostor_cohort_dim_matches CHECK (dim = 192)
);

-- 코호트는 모델별로 통째로 읽어 메모리에 적재한다. 모델이 바뀌면 코호트도
-- 다시 만들어야 하므로 model로 좁히는 인덱스만 둔다.
CREATE INDEX IF NOT EXISTS impostor_cohort_model_idx ON impostor_cohort (model);
