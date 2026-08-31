/**
 * 서버 관리자 API 응답 타입.
 *
 * 서버가 스키마를 확장해도 대시보드가 깨지지 않도록, 필수로 쓰는 필드만
 * 선언하고 나머지는 무시한다.
 */

export interface Overview {
  total_attempts: number;
  outcomes: { outcome: string; count: number }[];
  verified_count: number;
  rejected_count: number;
  audio_rejected_count: number;
  decision_count: number;
  /** 판정 중 통과 비율. **정확도가 아니다** — 아래 caveat 참조. */
  verified_ratio: number | null;
  latency: { avg_ms: number | null; p50_ms: number | null; p95_ms: number | null };
  enrollments: { total: number; active: number; users: number };
  cohort_size: number;
  first_attempt_at: string | null;
  last_attempt_at: string | null;
  caveat: string;
}

export interface TimeBucket {
  bucket: string;
  verified: number;
  rejected: number;
  audio_rejected: number;
  not_enrolled: number;
  other: number;
  total: number;
  avg_elapsed_ms: number | null;
}

export interface TimeSeries {
  hours: number;
  bucket_minutes: number;
  buckets: TimeBucket[];
}

export interface ScoreBin {
  lower: number;
  upper: number;
  verified: number;
  rejected: number;
  total: number;
}

export interface ScoreDistribution {
  field: string;
  threshold: number | null;
  sample_count: number;
  min_score: number | null;
  max_score: number | null;
  bins: ScoreBin[];
  caveat: string;
}

export interface ThresholdImpactRow {
  threshold: number;
  would_pass: number;
  would_fail: number;
  flipped_from_pass: number;
  flipped_from_fail: number;
  total: number;
}

export interface ThresholdImpact {
  field: string;
  impacts: ThresholdImpactRow[];
  caveat: string;
}

export interface Speaker {
  user_id: string;
  active_count: number;
  inactive_count: number;
  model: string | null;
  dim: number | null;
  last_enrolled_at: string | null;
  avg_speech_duration_sec: number | null;
  /** 현재 모델과 다른 모델로 등록됨 — 검증이 거부된다. */
  needs_reenrollment: boolean;
}

export interface Speakers {
  current_model: string;
  current_dim: number;
  needs_reenrollment_count: number;
  speakers: Speaker[];
}

export interface Attempt {
  id: number;
  user_id: string;
  outcome: string;
  is_verified: boolean | null;
  raw_cosine: number | null;
  normalized_score: number | null;
  match_probability: number | null;
  threshold: number | null;
  model: string | null;
  speech_duration_sec: number | null;
  error_code: string | null;
  client_ip: string | null;
  elapsed_ms: number | null;
  created_at: string;
}

export interface Attempts {
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  attempts: Attempt[];
}

export interface CalibrationMetrics {
  eer: number;
  eer_threshold: number;
  min_dcf: number;
  min_dcf_threshold: number;
  separation: number;
  n_genuine: number;
  n_impostor: number;
}

export interface CalibrationReport {
  backend?: string;
  model?: string;
  dataset: {
    eval_split: string;
    cohort_split: string;
    eval_speakers: number;
    trials: number;
    cohort_embeddings: number;
  };
  raw_cosine: CalibrationMetrics;
  as_norm: Record<string, CalibrationMetrics>;
  best_top_k: number;
}

export interface Calibration {
  active: {
    embedding_backend: string;
    embedding_model: string;
    embedding_dim: number;
    asnorm_enabled: boolean;
    asnorm_active: boolean;
    asnorm_top_k: number;
    asnorm_threshold: number;
    match_threshold: number;
    cohort_size: number;
  };
  report: CalibrationReport | null;
  how_to_apply: Record<string, string>;
}
