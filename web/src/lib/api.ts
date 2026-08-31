/**
 * 관리자 API 호출.
 *
 * 서버 사이드에서만 호출한다. 관리자 토큰이 브라우저로 나가면 안 되므로
 * `NEXT_PUBLIC_` 접두사를 쓰지 않으며, 모든 페이지가 서버 컴포넌트다.
 */

import type {
  Attempts,
  Calibration,
  Overview,
  ScoreDistribution,
  Speakers,
  Spoofing,
  ThresholdImpact,
  TimeSeries,
} from "./types";

const BASE_URL = process.env.VG_API_BASE_URL ?? "http://127.0.0.1:8000";
const ADMIN_TOKEN = process.env.VG_ADMIN_TOKEN ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  if (!ADMIN_TOKEN) {
    throw new ApiError(
      "VG_ADMIN_TOKEN이 설정되지 않았습니다. .env.local에 서버와 동일한 토큰을 넣어주세요.",
      0,
    );
  }

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/api/v1/admin${path}`, {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
      // 대시보드는 항상 최신 상태를 보여야 한다. 캐시된 통계는 오해를 부른다.
      cache: "no-store",
    });
  } catch {
    throw new ApiError(`분석 서버(${BASE_URL})에 연결할 수 없습니다.`, 0);
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // 본문이 JSON이 아니면 상태 코드만 쓴다
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export const api = {
  overview: (hours?: number) =>
    get<Overview>(`/overview${hours ? `?hours=${hours}` : ""}`),
  timeseries: (hours = 24, bucketMinutes = 60) =>
    get<TimeSeries>(`/timeseries?hours=${hours}&bucket_minutes=${bucketMinutes}`),
  scoreDistribution: (bins = 20) =>
    get<ScoreDistribution>(`/score-distribution?bins=${bins}`),
  thresholdImpact: (thresholds: number[]) =>
    get<ThresholdImpact>(`/threshold-impact?thresholds=${thresholds.join(",")}`),
  speakers: () => get<Speakers>("/speakers"),
  attempts: (limit = 50, offset = 0) =>
    get<Attempts>(`/attempts?limit=${limit}&offset=${offset}`),
  calibration: () => get<Calibration>("/calibration"),
  spoofing: () => get<Spoofing>("/spoofing"),
};
