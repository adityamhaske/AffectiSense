"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ResultsDashboard from "@/components/ResultsDashboard";

import { PredictionResult } from "@/types";

export default function ResultsPage() {
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("affectisense_result");
    if (stored) {
      setResult(JSON.parse(stored));
    }
  }, []);

  if (!result) {
    return (
      <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center px-6 text-center">
        <div className="text-6xl mb-6">📊</div>
        <h2 className="text-2xl font-bold mb-3">No Results Yet</h2>
        <p className="text-[var(--color-text-secondary)] mb-8 max-w-md">
          Record or upload audio/video on the Analyze page to see your screening results here.
        </p>
        <Link href="/analyze"
          className="px-8 py-3 rounded-xl font-semibold text-white transition-all hover:scale-105 glow-primary"
          style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
          Go to Analyze →
        </Link>
      </div>
    );
  }

  return (
    <div className="py-12 px-6">
      <div className="max-w-5xl mx-auto">
        <ResultsDashboard result={result} />
        <div className="flex justify-center gap-4 mt-10">
          <Link href="/analyze"
            className="px-6 py-3 rounded-xl font-semibold text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-primary)] transition-all hover:text-[var(--color-text-primary)]">
            ← New Recording
          </Link>
        </div>
      </div>
    </div>
  );
}
