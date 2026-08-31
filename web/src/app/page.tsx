/**
 * 대시보드 메인 (03 §3).
 *
 * API 호출 횟수, 평균 응답 시간, 판정 결과 분포를 보여준다.
 */

import { api, ApiError } from "@/lib/api";
import {
  Card,
  Caveat,
  ErrorPanel,
  formatMs,
  formatTime,
  OUTCOME_LABEL,
  outcomeTone,
  Badge,
  Stat,
} from "@/components/ui";
import type { Overview, ScoreDistribution, TimeSeries } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let overview: Overview;
  let timeseries: TimeSeries;
  let distribution: ScoreDistribution;

  try {
    [overview, timeseries, distribution] = await Promise.all([
      api.overview(),
      api.timeseries(24, 60),
      api.scoreDistribution(16),
    ]);
  } catch (e) {
    return <ErrorPanel message={e instanceof ApiError ? e.message : String(e)} />;
  }

  const latencyTone =
    overview.latency.p95_ms == null
      ? "default"
      : overview.latency.p95_ms > 2000
        ? "bad"
        : overview.latency.p95_ms > 800
          ? "warn"
          : "good";

  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="총 검증 시도"
          value={overview.total_attempts.toLocaleString()}
          hint={`최근: ${formatTime(overview.last_attempt_at)}`}
        />
        <Stat
          label="판정 통과율"
          value={
            overview.verified_ratio == null
              ? "—"
              : `${(overview.verified_ratio * 100).toFixed(1)}%`
          }
          hint={`판정 ${overview.decision_count}건 중 통과 ${overview.verified_count}건`}
        />
        <Stat
          label="응답 시간 p95"
          value={formatMs(overview.latency.p95_ms)}
          hint={`평균 ${formatMs(overview.latency.avg_ms)} · p50 ${formatMs(overview.latency.p50_ms)}`}
          tone={latencyTone}
        />
        <Stat
          label="등록 사용자"
          value={overview.enrollments.users.toLocaleString()}
          hint={`활성 성문 ${overview.enrollments.active}건 · 코호트 ${overview.cohort_size}`}
        />
      </div>

      <Card
        title="판정 결과 분포"
        subtitle="검증 시도가 어떻게 끝났는지 — 반려 사유별 비중을 보면 UX 문제를 찾을 수 있다"
      >
        <div className="space-y-2">
          {overview.outcomes.map((o) => {
            const pct = overview.total_attempts
              ? (o.count / overview.total_attempts) * 100
              : 0;
            return (
              <div key={o.outcome} className="flex items-center gap-3">
                <div className="w-28 shrink-0">
                  <Badge tone={outcomeTone(o.outcome)}>
                    {OUTCOME_LABEL[o.outcome] ?? o.outcome}
                  </Badge>
                </div>
                <div className="h-5 flex-1 overflow-hidden rounded bg-slate-100">
                  <div
                    className="h-full bg-slate-400"
                    style={{ width: `${Math.max(pct, 1)}%` }}
                  />
                </div>
                <span className="w-24 shrink-0 text-right text-sm tabular-nums text-slate-600">
                  {o.count}건 ({pct.toFixed(0)}%)
                </span>
              </div>
            );
          })}
        </div>
        <Caveat>{overview.caveat}</Caveat>
      </Card>

      <Card
        title="시간대별 요청 추이 (최근 24시간)"
        subtitle="요청량과 응답 시간을 함께 본다 — 부하가 지연으로 이어지는지 확인"
      >
        {timeseries.buckets.length === 0 ? (
          <p className="text-sm text-slate-500">최근 24시간 내 요청이 없습니다.</p>
        ) : (
          <TimeSeriesChart series={timeseries} />
        )}
      </Card>

      <Card
        title="판정 점수 분포"
        subtitle={
          distribution.sample_count > 0
            ? `${distribution.field} 기준 · 표본 ${distribution.sample_count}건 · 임계값 ${distribution.threshold?.toFixed(4) ?? "—"}`
            : "판정된 시도가 아직 없습니다"
        }
      >
        {distribution.sample_count === 0 ? (
          <p className="text-sm text-slate-500">
            검증이 누적되면 통과·거부 점수 분포가 여기에 표시됩니다.
          </p>
        ) : (
          <>
            <ScoreHistogram distribution={distribution} />
            <Caveat>{distribution.caveat}</Caveat>
          </>
        )}
      </Card>
    </>
  );
}

function TimeSeriesChart({ series }: { series: TimeSeries }) {
  const max = Math.max(...series.buckets.map((b) => b.total), 1);

  return (
    <div className="space-y-1">
      {series.buckets.map((b) => (
        <div key={b.bucket} className="flex items-center gap-3 text-xs">
          <span className="w-32 shrink-0 tabular-nums text-slate-500">
            {new Date(b.bucket).toLocaleString("ko-KR", {
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              hour12: false,
            })}
          </span>
          <div className="flex h-5 flex-1 overflow-hidden rounded bg-slate-100">
            {/* 결과별로 색을 나눠 쌓는다 — 요청량만 보면 실패가 늘어난 걸 놓친다 */}
            <Segment count={b.verified} max={max} className="bg-emerald-400" />
            <Segment count={b.rejected} max={max} className="bg-red-400" />
            <Segment count={b.audio_rejected} max={max} className="bg-amber-400" />
            <Segment count={b.not_enrolled} max={max} className="bg-slate-400" />
            <Segment count={b.other} max={max} className="bg-slate-300" />
          </div>
          <span className="w-14 shrink-0 text-right tabular-nums text-slate-600">
            {b.total}건
          </span>
          <span className="w-16 shrink-0 text-right tabular-nums text-slate-400">
            {formatMs(b.avg_elapsed_ms)}
          </span>
        </div>
      ))}
      <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500">
        <Legend className="bg-emerald-400" label="통과" />
        <Legend className="bg-red-400" label="거부" />
        <Legend className="bg-amber-400" label="오디오 반려" />
        <Legend className="bg-slate-400" label="미등록" />
      </div>
    </div>
  );
}

function Segment({
  count,
  max,
  className,
}: {
  count: number;
  max: number;
  className: string;
}) {
  if (count === 0) return null;
  return <div className={className} style={{ width: `${(count / max) * 100}%` }} />;
}

function Legend({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`inline-block h-2.5 w-2.5 rounded-sm ${className}`} />
      {label}
    </span>
  );
}

function ScoreHistogram({ distribution }: { distribution: ScoreDistribution }) {
  const max = Math.max(...distribution.bins.map((b) => b.total), 1);
  const threshold = distribution.threshold;

  return (
    <div className="space-y-1">
      {distribution.bins.map((b, i) => {
        // 임계값이 이 구간에 걸리면 표시한다 — 분포가 경계에서 갈리는지 눈으로 확인
        const holdsThreshold =
          threshold != null && threshold >= b.lower && threshold < b.upper;
        return (
          <div key={i} className="flex items-center gap-3 text-xs">
            <span className="w-24 shrink-0 text-right tabular-nums text-slate-500">
              {b.lower.toFixed(2)}
            </span>
            <div
              className={`flex h-5 flex-1 overflow-hidden rounded bg-slate-100 ${
                holdsThreshold ? "ring-2 ring-indigo-400" : ""
              }`}
            >
              <Segment count={b.verified} max={max} className="bg-emerald-400" />
              <Segment count={b.rejected} max={max} className="bg-red-400" />
            </div>
            <span className="w-20 shrink-0 text-right tabular-nums text-slate-500">
              {b.total > 0 ? `${b.total}건` : ""}
            </span>
          </div>
        );
      })}
      {threshold != null && (
        <p className="mt-2 text-xs text-indigo-600">
          ─ 표시된 구간이 현재 판정 임계값({threshold.toFixed(4)})을 포함합니다.
          통과(초록)와 거부(빨강)가 이 경계에서 깨끗이 갈릴수록 분리가 잘 된 것입니다.
        </p>
      )}
    </div>
  );
}
