/**
 * 임계값 캘리브레이션 (FR-11, FR-12).
 *
 * **핵심 원칙:** EER·minDCF는 레이블이 있는 오프라인 평가에서만 나온다. 운영
 * 로그에는 정답이 없어 오류율을 계산할 수 없으므로, 이 화면은 두 종류의 근거를
 * 명확히 구분해 보여준다.
 *
 *   1. 오프라인 캘리브레이션 (eval/) — EER·minDCF, 임계값 결정의 진짜 근거
 *   2. 운영 영향도 (감사 로그)   — 임계값을 바꾸면 과거 판정이 몇 건 뒤집히는가
 */

import { api, ApiError } from "@/lib/api";
import { Badge, Card, Caveat, ErrorPanel, Stat } from "@/components/ui";
import type { Calibration, ThresholdImpact } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function CalibrationPage() {
  let calibration: Calibration;
  let impact: ThresholdImpact | null = null;

  try {
    calibration = await api.calibration();
  } catch (e) {
    return <ErrorPanel message={e instanceof ApiError ? e.message : String(e)} />;
  }

  // 현재 임계값 주변을 후보로 잡아 영향도를 본다
  const current = calibration.active.asnorm_threshold;
  const candidates = [
    current - 2,
    current - 1,
    current,
    current + 1,
    current + 2,
    current + 4,
  ].map((v) => Number(v.toFixed(4)));

  try {
    impact = await api.thresholdImpact(candidates);
  } catch {
    // 영향도는 부가 정보다. 실패해도 캘리브레이션 결과는 보여준다.
  }

  const { active, report } = calibration;

  return (
    <>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat
          label="AS-Norm 임계값"
          value={active.asnorm_threshold.toFixed(4)}
          hint={`K=${active.asnorm_top_k} · 코호트 ${active.cohort_size}`}
        />
        <Stat
          label="원시 코사인 임계값"
          value={active.match_threshold.toFixed(4)}
          hint="AS-Norm 폴백 시 사용"
        />
        <Stat
          label="정규화 상태"
          value={active.asnorm_active ? "적용 중" : "꺼짐"}
          tone={active.asnorm_active ? "good" : "bad"}
          hint={active.asnorm_active ? undefined : "코호트 미적재 — 판정 신뢰도 낮음"}
        />
        <Stat
          label="임베딩"
          value={`${active.embedding_dim}차원`}
          hint={active.embedding_model.split("/").pop() ?? active.embedding_model}
        />
      </div>

      {!active.asnorm_active && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <strong className="font-semibold">AS-Norm이 적용되지 않고 있습니다.</strong>{" "}
          임포스터 코호트가 비어 있어 원시 코사인으로 판정 중입니다. 점수 편차가
          보정되지 않아 실패율이 높아집니다.{" "}
          <code className="rounded bg-red-100 px-1">
            python -m eval.seed_cohort --replace
          </code>{" "}
          로 코호트를 적재하세요.
        </div>
      )}

      <Card
        title="오프라인 캘리브레이션 결과"
        subtitle="레이블이 있는 평가 데이터에서 산출 — 임계값 결정의 유일한 정당한 근거"
      >
        {!report ? (
          <div className="text-sm text-slate-600">
            <p>캘리브레이션 보고서가 없습니다.</p>
            <p className="mt-2 text-xs text-slate-500">
              <code className="rounded bg-slate-100 px-1">python -m eval.calibrate</code>{" "}
              를 실행하면 현재 백엔드에 대한 EER·minDCF가 산출됩니다. 그전까지 임계값은
              근거 없는 값이며 판정 결과를 신뢰해서는 안 됩니다.
            </p>
          </div>
        ) : (
          <>
            <p className="mb-3 text-xs text-slate-500">
              {report.dataset.eval_split} {report.dataset.eval_speakers}화자 ·{" "}
              {report.dataset.trials.toLocaleString()}트라이얼 · 코호트{" "}
              {report.dataset.cohort_split} {report.dataset.cohort_embeddings}개
              <span className="ml-2 text-slate-400">(평가·코호트 화자는 분리됨)</span>
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                  <th className="px-2 py-2 font-medium">방식</th>
                  <th className="px-2 py-2 text-right font-medium">EER</th>
                  <th className="px-2 py-2 text-right font-medium">EER 임계값</th>
                  <th className="px-2 py-2 text-right font-medium">minDCF</th>
                  <th className="px-2 py-2 text-right font-medium">minDCF 임계값</th>
                  <th className="px-2 py-2 text-right font-medium">분리도</th>
                  <th className="px-2 py-2 font-medium" />
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-slate-100">
                  <td className="px-2 py-2">원시 코사인</td>
                  <td className="px-2 py-2 text-right tabular-nums">
                    {(report.raw_cosine.eer * 100).toFixed(2)}%
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums">
                    {report.raw_cosine.eer_threshold.toFixed(4)}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums">
                    {report.raw_cosine.min_dcf.toFixed(4)}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                    {report.raw_cosine.min_dcf_threshold.toFixed(4)}
                  </td>
                  <td className="px-2 py-2 text-right tabular-nums text-slate-500">
                    {report.raw_cosine.separation.toFixed(2)}
                  </td>
                  <td className="px-2 py-2">
                    {active.match_threshold.toFixed(4) ===
                      report.raw_cosine.eer_threshold.toFixed(4) && (
                      <Badge tone="good">적용 중</Badge>
                    )}
                  </td>
                </tr>
                {Object.entries(report.as_norm).map(([k, m]) => {
                  const isActive =
                    Number(k) === active.asnorm_top_k &&
                    m.eer_threshold.toFixed(4) === active.asnorm_threshold.toFixed(4);
                  return (
                    <tr key={k} className="border-b border-slate-100">
                      <td className="px-2 py-2">AS-Norm (K={k})</td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {(m.eer * 100).toFixed(2)}%
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {m.eer_threshold.toFixed(4)}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {m.min_dcf.toFixed(4)}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                        {m.min_dcf_threshold.toFixed(4)}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-slate-500">
                        {m.separation.toFixed(2)}
                      </td>
                      <td className="px-2 py-2">
                        {isActive && <Badge tone="good">적용 중</Badge>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-slate-500">
              EER은 오수락·오거부를 대칭으로 볼 때의 기준이고, minDCF는 사칭이 드문
              실제 환경을 반영해 오수락에 더 큰 가중을 둡니다. 인증 시스템에서는
              minDCF 쪽이 대체로 더 적합한 기준입니다.
            </p>
          </>
        )}
      </Card>

      {impact && impact.impacts.length > 0 && (
        <Card
          title="임계값 변경 영향도 (운영 로그 기준)"
          subtitle="임계값을 바꾸면 과거 판정이 몇 건 뒤집히는가"
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs text-slate-500">
                <th className="px-2 py-2 font-medium">후보 임계값</th>
                <th className="px-2 py-2 text-right font-medium">통과</th>
                <th className="px-2 py-2 text-right font-medium">거부</th>
                <th className="px-2 py-2 text-right font-medium">통과 → 거부</th>
                <th className="px-2 py-2 text-right font-medium">거부 → 통과</th>
              </tr>
            </thead>
            <tbody>
              {impact.impacts.map((row) => {
                const isCurrent = row.threshold.toFixed(4) === current.toFixed(4);
                return (
                  <tr
                    key={row.threshold}
                    className={`border-b border-slate-100 ${isCurrent ? "bg-indigo-50" : ""}`}
                  >
                    <td className="px-2 py-2 tabular-nums">
                      {row.threshold.toFixed(4)}
                      {isCurrent && <span className="ml-2 text-xs text-indigo-600">현재</span>}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums">{row.would_pass}</td>
                    <td className="px-2 py-2 text-right tabular-nums">{row.would_fail}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-red-600">
                      {row.flipped_from_pass || "—"}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-amber-600">
                      {row.flipped_from_fail || "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <Caveat>{impact.caveat}</Caveat>
        </Card>
      )}

      <Card
        title="임계값 적용 방법"
        subtitle="런타임 변경 API는 제공하지 않는다 — 인증 파라미터를 원격에서 바꾸는 것 자체가 공격면이다"
      >
        <pre className="overflow-x-auto rounded bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">
          {`# server/.env 또는 배포 환경변수
VG_ASNORM_THRESHOLD=${active.asnorm_threshold}
VG_ASNORM_TOP_K=${active.asnorm_top_k}
VG_MATCH_THRESHOLD=${active.match_threshold}

# 변경 후 서버 재시작`}
        </pre>
        <p className="mt-3 text-xs text-slate-500">
          정규화 점수와 원시 코사인은 척도가 완전히 다릅니다(정규화 점수는 코호트
          표준편차 단위라 [-1, 1]에 갇히지 않습니다). 두 임계값을 서로 바꿔 넣으면 안
          됩니다.
        </p>
      </Card>
    </>
  );
}
