import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "VoiceGuard 관리자",
  description: "화자 인증 시스템 운영 대시보드",
};

const NAV = [
  { href: "/", label: "대시보드" },
  { href: "/speakers", label: "화자 DB" },
  { href: "/attempts", label: "오딧 트레일" },
  { href: "/spoofing", label: "스푸핑" },
  { href: "/calibration", label: "캘리브레이션" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <div className="mx-auto max-w-6xl px-6 py-8">
          <header className="mb-8 flex flex-wrap items-baseline justify-between gap-4 border-b border-slate-200 pb-4">
            <div>
              <h1 className="text-xl font-bold">VoiceGuard 관리자</h1>
              <p className="mt-1 text-xs text-slate-500">
                화자 인증 시스템 운영 대시보드
              </p>
            </div>
            <nav className="flex gap-1">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded px-3 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-200 hover:text-slate-900"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </header>
          <main className="space-y-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
