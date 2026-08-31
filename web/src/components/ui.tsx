/** 대시보드 공용 컴포넌트. */

import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-slate-200 bg-white p-5 shadow-sm ${className}`}
    >
      {title && (
        <header className="mb-4">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneClass = {
    default: "text-slate-900",
    good: "text-emerald-600",
    warn: "text-amber-600",
    bad: "text-red-600",
  }[tone];

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={`mt-1 text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-400">{hint}</p>}
    </div>
  );
}

/**
 * 지표 해석 주의사항.
 *
 * 서버가 응답에 실어 보내는 caveat을 그대로 노출한다. 숫자만 보고 오해하는 것을
 * 막는 것이 이 대시보드에서 가장 중요한 일이다.
 */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900">
      {children}
    </p>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const toneClass = {
    neutral: "bg-slate-100 text-slate-700",
    good: "bg-emerald-100 text-emerald-800",
    warn: "bg-amber-100 text-amber-800",
    bad: "bg-red-100 text-red-800",
  }[tone];

  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${toneClass}`}>
      {children}
    </span>
  );
}

/** 서버 연결 실패 안내. 대시보드는 서버 없이 아무것도 보여줄 수 없다. */
export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-5">
      <h2 className="text-sm font-semibold text-red-900">데이터를 불러오지 못했습니다</h2>
      <p className="mt-2 text-sm text-red-800">{message}</p>
      <ul className="mt-3 list-inside list-disc text-xs text-red-700">
        <li>분석 서버가 실행 중인지 확인 (server/README.md)</li>
        <li>서버가 PostgreSQL로 기동됐는지 확인 — 인메모리로는 통계를 제공하지 않는다</li>
        <li>web/.env.local의 VG_ADMIN_TOKEN이 서버의 VG_ADMIN_TOKEN과 같은지 확인</li>
      </ul>
    </div>
  );
}

export function outcomeTone(outcome: string): "neutral" | "good" | "warn" | "bad" {
  switch (outcome) {
    case "verified":
      return "good";
    case "rejected":
      return "bad";
    case "audio_rejected":
    case "not_enrolled":
      return "warn";
    default:
      return "neutral";
  }
}

export const OUTCOME_LABEL: Record<string, string> = {
  verified: "통과",
  rejected: "거부",
  audio_rejected: "오디오 반려",
  not_enrolled: "미등록",
  model_mismatch: "모델 불일치",
};

export function formatMs(ms: number | null): string {
  return ms == null ? "—" : `${ms.toFixed(0)}ms`;
}

export function formatScore(v: number | null, digits = 3): string {
  return v == null ? "—" : v.toFixed(digits);
}

export function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("ko-KR", { hour12: false });
}
