import type {Metadata} from "next";
import {headers} from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "learn-verl.banghaochi.com";
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto");
  const protocol = forwardedProtocol ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "verl 深度学习手册（非官方）";
  const description = "从 RL 基础到当前 V1 源码数据流的系统中文教程。";

  return {
    metadataBase: new URL(origin),
    title,
    description,
    alternates: {canonical: `${origin}/guide/`},
    icons: {
      icon: "/guide/img/verl-logo.png",
      shortcut: "/guide/img/verl-logo.png",
    },
    openGraph: {
      type: "website",
      locale: "zh_CN",
      url: `${origin}/guide/`,
      title,
      description,
      images: [{url: `${origin}/guide/img/og.png`, width: 1731, height: 909}],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${origin}/guide/img/og.png`],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
