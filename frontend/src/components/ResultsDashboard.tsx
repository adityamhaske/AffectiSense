"use client";

import { PredictionResult } from "@/app/page";

interface Props {
  result: PredictionResult;
}

function ConfidenceRing({ value, size = 120, stroke = 8, label }: {
  value: number; size?: number; stroke?: number; label: string;
}) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - value * circumference;
  const color = value > 0.7 ? "#10b981" : value > 0.4 ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size}>
        <circle cx={size/2} cy={size/2} r={radius} fill="none"
          stroke="var(--color-border)" strokeWidth={stroke} />
        <circle cx={size/2} cy={size/2} r={radius} fill="none"
          stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          className="progress-ring-fill" />
        <text x={size/2} y={size/2 - 6} textAnchor="middle" fill="var(--color-text-primary)"
          fontSize="22" fontWeight="700">{Math.round(value * 100)}%</text>
        <text x={size/2} y={size/2 + 14} textAnchor="middle" fill="var(--color-text-muted)"
          fontSize="10" fontWeight="500">{label}</text>
      </svg>
    </div>
  );
}

function SeverityBar({ level, score }: { level: string; score: number }) {
  const segments = ["none", "minimal", "mild", "moderate", "moderately_severe", "severe"];
  const colors = ["#10b981", "#34d399", "#fbbf24", "#f59e0b", "#ef4444", "#dc2626"];
  const activeIdx = segments.indexOf(level);

  return (
    <div>
      <div className="flex justify-between text-xs text-[var(--color-text-muted)] mb-2">
        <span>None</span><span>Severe</span>
      </div>
      <div className="flex gap-1 h-3 rounded-full overflow-hidden">
        {segments.map((seg, i) => (
          <div key={seg} className="flex-1 rounded-sm transition-all duration-700"
            style={{
              background: i <= activeIdx ? colors[i] : "var(--color-border)",
              opacity: i <= activeIdx ? 1 : 0.3,
            }} />
        ))}
      </div>
      <div className="text-center mt-2">
        <span className="text-sm font-semibold capitalize" style={{ color: colors[activeIdx] || "#94a3b8" }}>
          {level.replace("_", " ")}
        </span>
        <span className="text-xs text-[var(--color-text-muted)] ml-2">({(score * 100).toFixed(0)}%)</span>
      </div>
    </div>
  );
}

export default function ResultsDashboard({ result }: Props) {
  const isDepressed = result.prediction === "depressed";

  return (
    <section className="py-16 px-6 animate-fade-in-up">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center">
          Analysis <span className="gradient-text">Results</span>
        </h2>

        {/* Top row: Prediction + Confidence + Severity */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {/* Prediction card */}
          <div className={`glass-card p-6 text-center ${isDepressed ? "glow-danger" : "glow-success"}`}>
            <div className={`w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center ${
              isDepressed ? "bg-[rgba(239,68,68,0.15)]" : "bg-[rgba(16,185,129,0.15)]"
            }`}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                stroke={isDepressed ? "#ef4444" : "#10b981"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                {isDepressed ? (
                  <><circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></>
                ) : (
                  <><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></>
                )}
              </svg>
            </div>
            <div className={`text-2xl font-bold capitalize ${isDepressed ? "text-[var(--color-danger)]" : "text-[var(--color-success)]"}`}>
              {result.prediction}
            </div>
            <div className="text-sm text-[var(--color-text-muted)] mt-1">
              Probability: {(result.depression_probability * 100).toFixed(1)}%
            </div>
          </div>

          {/* Confidence ring */}
          <div className="glass-card p-6 flex flex-col items-center justify-center">
            <ConfidenceRing value={result.overall_confidence} label="Confidence" />
            <div className="text-xs text-[var(--color-text-muted)] mt-3">
              {result.modality_completeness === 1 ? "Full modality coverage" : "Partial modality coverage"}
            </div>
          </div>

          {/* Severity */}
          <div className="glass-card p-6 flex flex-col justify-center">
            <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-4 text-center">Severity Assessment</h3>
            <SeverityBar level={result.severity_level} score={result.severity_score} />
          </div>
        </div>

        {/* Per-modality breakdown */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {result.modality_scores.map((ms) => (
            <div key={ms.modality} className={`glass-card p-6 ${!ms.available ? "opacity-50" : ""}`}>
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                  ms.modality === "audio" ? "bg-[rgba(99,102,241,0.15)]" : "bg-[rgba(6,182,212,0.15)]"
                }`}>
                  {ms.modality === "audio" ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    </svg>
                  ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/>
                    </svg>
                  )}
                </div>
                <div>
                  <h3 className="font-semibold capitalize">{ms.modality} Analysis</h3>
                  <span className={`text-xs ${ms.available ? "text-[var(--color-success)]" : "text-[var(--color-text-muted)]"}`}>
                    {ms.available ? "✓ Active" : "— Not provided"}
                  </span>
                </div>
                {ms.prediction_score !== null && (
                  <div className="ml-auto text-right">
                    <div className="text-lg font-bold">{(ms.prediction_score * 100).toFixed(1)}%</div>
                    <div className="text-xs text-[var(--color-text-muted)]">score</div>
                  </div>
                )}
              </div>
              {ms.key_indicators.length > 0 && (
                <ul className="space-y-1.5">
                  {ms.key_indicators.map((ind, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-text-secondary)]">
                      <span className="text-[var(--color-warning)] mt-0.5">⚠</span> {ind}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>

        {/* Clinical summary */}
        <div className="glass-card p-6 mb-8">
          <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-3 flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
            </svg>
            Clinical Summary
          </h3>
          <p className="text-[var(--color-text-primary)] leading-relaxed">{result.clinical_summary}</p>
        </div>

        {/* Risk / Protective factors */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-[var(--color-danger)] mb-3">⚠ Risk Factors</h3>
            <ul className="space-y-2">
              {result.risk_factors.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-text-secondary)]">
                  <span className="text-[var(--color-danger)] mt-0.5">•</span> {f}
                </li>
              ))}
            </ul>
          </div>
          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-[var(--color-success)] mb-3">✓ Protective Factors</h3>
            <ul className="space-y-2">
              {result.protective_factors.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-text-secondary)]">
                  <span className="text-[var(--color-success)] mt-0.5">•</span> {f}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Metadata bar */}
        <div className="flex flex-wrap justify-center gap-6 text-xs text-[var(--color-text-muted)]">
          <span>Model: v{result.model_version}</span>
          <span>•</span>
          <span>Processing: {result.processing_time_ms.toFixed(0)}ms</span>
          <span>•</span>
          <span>Modalities: {result.modalities_used.join(", ")}</span>
        </div>
      </div>
    </section>
  );
}
