-- Phase 6: 임베딩 차원 고정 해제
--
-- 백엔드를 SpeechBrain ECAPA-TDNN(192차원)에서 WeSpeaker ResNet34-LM(256차원)으로
-- 전환하면서 vector(192) 제약이 걸림돌이 됐다. Phase 7에서 ERes2NetV2로 또
-- 바꿀 예정이므로, 차원을 바꿀 때마다 테이블을 새로 만드는 대신 컬럼을
-- 차원 무제약 `vector`로 바꾼다.
--
-- 이렇게 해도 되는 이유: 1:1 검증은 user_id로 좁힌 뒤 소수 벡터만 애플리케이션에서
-- 비교하므로 ANN 인덱스가 필요 없다. pgvector는 차원 무제약 vector 컬럼에
-- 인덱스를 만들 수 없을 뿐, 저장과 연산자는 정상 동작한다.
--
-- 1:N 식별을 도입해 HNSW가 필요해지면, 그때는 차원별 부분 인덱스나 차원별
-- 파티션으로 나눠야 한다. 그 시점까지 미룬다.
--
-- 차원이 섞여 저장되므로 **비교 전 dim/model 일치 확인이 필수**다. 서버는 이미
-- model_mismatch로 거부한다 (02 §6).

ALTER TABLE speaker_enrollments DROP CONSTRAINT IF EXISTS speaker_enrollments_dim_matches;
ALTER TABLE speaker_enrollments ALTER COLUMN embedding TYPE vector;

ALTER TABLE impostor_cohort DROP CONSTRAINT IF EXISTS impostor_cohort_dim_matches;
ALTER TABLE impostor_cohort ALTER COLUMN embedding TYPE vector;

-- dim 컬럼은 그대로 둔다. 차원 제약이 사라진 만큼 메타데이터의 중요성이 커졌다.
ALTER TABLE speaker_enrollments ADD CONSTRAINT speaker_enrollments_dim_positive
    CHECK (dim > 0);
ALTER TABLE impostor_cohort ADD CONSTRAINT impostor_cohort_dim_positive
    CHECK (dim > 0);
