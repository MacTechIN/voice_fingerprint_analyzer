/**
 * 화자 데이터베이스 (03 §3).
 *
 * 원본 오디오는 분석 후 즉시 파기되고 벡터도 노출하지 않는다. 관리자가 알아야
 * 할 것은 **어떤 모델로 언제 등록됐고 재등록이 필요한가**이다.
 */

import { api, ApiError } from "@/lib/api";
import { Badge, Card, ErrorPanel, formatTime, Stat } from "@/components/ui";
import type { Speakers } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function SpeakersPage() {
  let data: Speakers;
  try {
    data = await api.speakers();
  } catch (e) {
    return <ErrorPanel message={e instanceof ApiError ? e.message : String(e)} />;
  }

  const stale = data.needs_reenrollment_count;

  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        <Stat label="등록 사용자" value={data.speakers.length.toLocaleString()} />
        <Stat
          label="재등록 필요"
          value={stale.toLocaleString()}
          tone={stale > 0 ? "warn" : "good"}
          hint={stale > 0 ? "검증이 model_mismatch로 거부됩니다" : "모두 현재 모델과 호환"}
        />
        <Stat
          label="현재 모델"
          value={`${data.current_dim}차원`}
          hint={data.current_model.split("/").pop() ?? data.current_model}
        />
      </div>

      {stale > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <strong className="font-semibold">재등록이 필요한 사용자가 있습니다.</strong>{" "}
          임베딩 모델을 교체하면 기존 벡터와 호환되지 않습니다. 해당 사용자의 검증
          요청은 <code className="rounded bg-amber-100 px-1">model_mismatch</code>로
          거부되며, 앱은 &ldquo;다시 등록해주세요&rdquo; 안내를 띄웁니다.
        </div>
      )}

      <Card
        title="등록 현황"
        subtitle="비활성 성문은 재등록으로 대체된 이력입니다 (삭제하지 않고 보존)"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                <th className="px-2 py-2 font-medium">사용자 ID</th>
                <th className="px-2 py-2 font-medium">상태</th>
                <th className="px-2 py-2 text-right font-medium">활성</th>
                <th className="px-2 py-2 text-right font-medium">비활성</th>
                <th className="px-2 py-2 font-medium">모델 / 차원</th>
                <th className="px-2 py-2 text-right font-medium">평균 발화</th>
                <th className="px-2 py-2 font-medium">최근 등록</th>
              </tr>
            </thead>
            <tbody>
              {data.speakers.map((s) => (
                <tr key={s.user_id} className="border-b border-slate-100">
                  <td className="px-2 py-2 font-mono text-xs">{s.user_id}</td>
                  <td className="px-2 py-2">
                    {s.needs_reenrollment ? (
                      <Badge tone="warn">재등록 필요</Badge>
                    ) : (
                      <Badge tone="good">정상</Badge>
                    )}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums">{s.active_count}</td>
                  <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                    {s.inactive_count}
                  </td>
                  <td className="px-2 py-2 text-xs text-slate-600">
                    {s.model ? `${s.model.split("/").pop()} / ${s.dim}` : "—"}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-slate-600">
                    {s.avg_speech_duration_sec != null
                      ? `${s.avg_speech_duration_sec.toFixed(1)}초`
                      : "—"}
                  </td>
                  <td className="px-2 py-2 text-xs text-slate-500">
                    {formatTime(s.last_enrolled_at)}
                  </td>
                </tr>
              ))}
              {data.speakers.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-2 py-6 text-center text-slate-500">
                    등록된 화자가 없습니다.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-400">
          보안을 위해 원본 오디오는 분석 직후 서버 메모리에서 파기되며, 성문 벡터도
          이 화면에 노출하지 않습니다.
        </p>
      </Card>
    </>
  );
}
