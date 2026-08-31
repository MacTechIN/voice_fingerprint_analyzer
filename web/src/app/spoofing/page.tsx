/**
 * 스푸핑 차단 감사 화면 (03 §5, FR-17).
 *
 * 합성 음성 공격 시도의 패턴(IP·계정·시간대)을 확인하고 급증을 알아챈다.
 */

import Link from "next/link";

import { api, ApiError } from "@/lib/api";
import { Badge, Card, Caveat, ErrorPanel, formatTime, Stat } from "@/components/ui";
import type { Spoofing } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function SpoofingPage() {
  let data: Spoofing;
  try {
    data = await api.spoofing();
  } catch (e) {
    return <ErrorPanel message={e instanceof ApiError ? e.message : String(e)} />;
  }

  return (
    <>
      {!data.active && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <strong className="font-semibold">딥페이크 탐지가 꺼져 있습니다.</strong>{" "}
          잘 만든 합성 음성은 화자 인증을 그대로 통과합니다 — 그것이 공격의 목적입니다.
          보안이 필요한 배포에서는{" "}
          <code className="rounded bg-red-100 px-1">VG_ANTISPOOF_ENABLED=true</code>
          로 켜야 합니다.
        </div>
      )}

      {data.surge_alert && (
        <div className="rounded-lg border border-orange-300 bg-orange-50 p-4 text-sm text-orange-900">
          <strong className="font-semibold">⚠ 차단 급증</strong> — 최근 24시간 차단이
          전체의 절반을 넘었습니다. 아래 IP·계정 분포를 확인하세요.
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="총 차단"
          value={data.total_blocked.toLocaleString()}
          hint={`최근: ${formatTime(data.last_blocked_at)}`}
          tone={data.total_blocked > 0 ? "warn" : "good"}
        />
        <Stat
          label="최근 24시간"
          value={data.blocked_24h.toLocaleString()}
          tone={data.surge_alert ? "bad" : "default"}
        />
        <Stat label="관련 IP" value={data.distinct_ips.toLocaleString()} />
        <Stat
          label="탐지 임계값"
          value={data.threshold.toFixed(2)}
          hint={
            data.mean_spoof_score != null
              ? `차단된 평균 점수 ${data.mean_spoof_score.toFixed(3)}`
              : undefined
          }
          tone={data.active ? "good" : "bad"}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="차단 상위 IP" subtitle="같은 IP에서 반복되면 자동화 공격 신호다">
          <RankTable
            rows={data.top_ips.map((r) => ({ label: r.client_ip, count: r.count }))}
            emptyMessage="차단된 시도가 없습니다."
          />
        </Card>

        <Card
          title="차단 상위 계정"
          subtitle="특정 계정에 집중되면 그 사용자가 표적일 수 있다"
        >
          <RankTable
            rows={data.top_users.map((r) => ({ label: r.user_id, count: r.count }))}
            emptyMessage="차단된 시도가 없습니다."
          />
        </Card>
      </div>

      <Card title="해석 시 주의">
        <Caveat>{data.caveat}</Caveat>
        <p className="mt-3 text-xs text-slate-500">
          차단 이력의 상세 점수는{" "}
          <Link href="/attempts" className="text-indigo-600 hover:underline">
            오딧 트레일
          </Link>
          의 스푸핑 열에서 확인할 수 있습니다. 탐지 성능(EER)은 레이블이 있는
          오프라인 평가(<code className="rounded bg-slate-100 px-1">eval/antispoof_eval.py</code>)
          에서만 나옵니다 — 운영 로그에는 &ldquo;놓친 위조&rdquo;가 기록되지 않기 때문입니다.
        </p>
      </Card>
    </>
  );
}

function RankTable({
  rows,
  emptyMessage,
}: {
  rows: { label: string; count: number }[];
  emptyMessage: string;
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-slate-500">{emptyMessage}</p>;
  }

  const max = Math.max(...rows.map((r) => r.count), 1);

  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-3 text-sm">
          <span className="w-40 shrink-0 truncate font-mono text-xs">{r.label}</span>
          <div className="h-4 flex-1 overflow-hidden rounded bg-slate-100">
            <div
              className="h-full bg-red-400"
              style={{ width: `${(r.count / max) * 100}%` }}
            />
          </div>
          <Badge tone="bad">{r.count}건</Badge>
        </div>
      ))}
    </div>
  );
}
