-- Phase 8: 딥페이크 탐지 점수 기록
--
-- 스푸핑 시도 패턴(IP·계정·시간대)을 사후 분석하려면 점수가 남아야 한다
-- (03 §5, FR-17). 차단된 시도는 outcome='spoof_detected'로 기록된다.

ALTER TABLE verification_attempts ADD COLUMN IF NOT EXISTS spoof_score REAL;

-- 스푸핑 시도만 빠르게 조회하기 위한 부분 인덱스. 전체 로그 대비 극소수일
-- 것이므로 부분 인덱스가 적합하다.
CREATE INDEX IF NOT EXISTS verification_attempts_spoof_idx
    ON verification_attempts (created_at DESC)
    WHERE outcome = 'spoof_detected';
