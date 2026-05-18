import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AffectiSense — Multimodal Mental Health Assessment",
  description:
    "AI-powered, modality-resilient depression screening platform using vocal biomarkers and facial expression analysis with calibrated confidence scores.",
  keywords: [
    "depression screening",
    "mental health AI",
    "multimodal analysis",
    "vocal biomarkers",
    "facial expression analysis",
    "clinical decision support",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
