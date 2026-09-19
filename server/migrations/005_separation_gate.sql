-- Phase 7 후속: 분리 게이트 결정 기록
--
-- 게이트 임계값은 배포 환경마다 다시 정해야 한다. 그러려면 실제 요청에서
-- **분리를 얼마나 건너뛰었는지, 그때 점수가 얼마였는지**가 남아야 한다.
-- Phase 4에서 겪은 것과 같은 함정이다 — 운영 로그에 없는 값은 사후에
-- 계산할 수 없고, 임계값은 영원히 LibriSpeech 기준으로 남는다.
--
-- 두 컬럼 모두 NULL을 허용한다. 분리를 끈 배포에서는 값이 없는 것이 정상이며,
-- 게이트 없이 항상 분리하는 경우 gate_score가 NULL이다.

ALTER TABLE verification_attempts
    ADD COLUMN IF NOT EXISTS separation_applied BOOLEAN;
ALTER TABLE verification_attempts
    ADD COLUMN IF NOT EXISTS separation_gate_score REAL;

-- 게이트 튜닝은 "분리를 거친 요청"과 "건너뛴 요청"의 점수 분포를 비교하는
-- 일이므로, 분리를 켠 요청만 추리는 부분 인덱스를 둔다.
CREATE INDEX IF NOT EXISTS verification_attempts_separation_idx
    ON verification_attempts (created_at DESC)
    WHERE separation_applied IS NOT NULL;
