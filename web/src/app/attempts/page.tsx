/**
 * 오딧 트레일 (03 §3).
 *
 * 원시 점수와 정규화 점수를 함께 보여준다 — 정규화가 실제로 어떤 효과를 냈는지
 * 사후에 추적할 수 있어야 하기 때문이다.
 */

import Link from "next/link";

import { api, ApiError } from "@/lib/api";
import {
  Badge,
  Card,
  ErrorPanel,
  formatMs,
  formatScore,
  formatTime,
  OUTCOME_LABEL,
  outcomeTone,
} from "@/components/ui";
import type { Attempts } from "@/lib/types";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 50;

export default async function AttemptsPage({
  searchParams,
}: {
  searchParams: Promise<{ offset?: string }>;
}) {
  const params = await searchParams;
  const offset = Math.max(0, Number(params.offset ?? 0) || 0);

  let data: Attempts;
  try {
    data = await api.attempts(PAGE_SIZE, offset);
  } catch (e) {
    return <ErrorPanel message={e instanceof ApiError ? e.message : String(e)} />;
  }

  return (
    <Card
      title="검증 시도 로그"
      subtitle={`전체 ${data.total.toLocaleString()}건 · ${offset + 1}–${offset + data.attempts.length} 표시`}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
              <th className="px-2 py-2 font-medium">시각</th>
              <th className="px-2 py-2 font-medium">사용자</th>
              <th className="px-2 py-2 font-medium">결과</th>
              <th className="px-2 py-2 text-right font-medium">원시 코사인</th>
              <th className="px-2 py-2 text-right font-medium">AS-Norm</th>
              <th className="px-2 py-2 text-right font-medium">임계값</th>
              <th className="px-2 py-2 text-right font-medium">일치도</th>
              <th className="px-2 py-2 text-right font-medium">발화</th>
              <th className="px-2 py-2 font-medium">IP</th>
              <th className="px-2 py-2 text-right font-medium">응답</th>
            </tr>
          </thead>
          <tbody>
            {data.attempts.map((a) => (
              <tr key={a.id} className="border-b border-slate-100">
                <td className="whitespace-nowrap px-2 py-2 text-xs text-slate-500">
                  {formatTime(a.created_at)}
                </td>
                <td className="px-2 py-2 font-mono text-xs">{a.user_id}</td>
                <td className="px-2 py-2">
                  <Badge tone={outcomeTone(a.outcome)}>
                    {OUTCOME_LABEL[a.outcome] ?? a.outcome}
                  </Badge>
                  {a.error_code && (
                    <span className="ml-1 font-mono text-[10px] text-slate-400">
                      {a.error_code}
                    </span>
                  )}
                </td>
                <td className="px-2 py-2 text-right tabular-nums">
                  {formatScore(a.raw_cosine, 4)}
                </td>
                <td className="px-2 py-2 text-right tabular-nums font-medium">
                  {formatScore(a.normalized_score, 4)}
                </td>
                <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                  {formatScore(a.threshold, 4)}
                </td>
                <td className="px-2 py-2 text-right tabular-nums">
                  {a.match_probability != null ? `${a.match_probability.toFixed(1)}%` : "—"}
                </td>
                <td className="px-2 py-2 text-right tabular-nums text-slate-500">
                  {a.speech_duration_sec != null
                    ? `${a.speech_duration_sec.toFixed(1)}초`
                    : "—"}
                </td>
                <td className="px-2 py-2 font-mono text-xs text-slate-500">
                  {a.client_ip ?? "—"}
                </td>
                <td className="px-2 py-2 text-right tabular-nums text-slate-500">
                  {formatMs(a.elapsed_ms)}
                </td>
              </tr>
            ))}
            {data.attempts.length === 0 && (
              <tr>
                <td colSpan={10} className="px-2 py-6 text-center text-slate-500">
                  검증 시도 기록이 없습니다.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <nav className="mt-4 flex items-center justify-between text-sm">
        {offset > 0 ? (
          <Link
            href={`/attempts?offset=${Math.max(0, offset - PAGE_SIZE)}`}
            className="rounded border border-slate-300 px-3 py-1.5 hover:bg-slate-100"
          >
            ← 이전
          </Link>
        ) : (
          <span />
        )}
        {data.has_more ? (
          <Link
            href={`/attempts?offset=${offset + PAGE_SIZE}`}
            className="rounded border border-slate-300 px-3 py-1.5 hover:bg-slate-100"
          >
            다음 →
          </Link>
        ) : (
          <span />
        )}
      </nav>

      <p className="mt-3 text-xs text-slate-400">
        AS-Norm 열이 비어 있으면 서버가 코호트 없이 원시 코사인으로 판정한 것입니다.
        그 시점의 판정은 점수 편차 보정이 적용되지 않아 신뢰도가 낮습니다.
      </p>
    </Card>
  );
}
