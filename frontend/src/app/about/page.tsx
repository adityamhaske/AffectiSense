import Link from "next/link";
import FeaturesSection from "@/components/FeaturesSection";

export default function AboutPage() {
  return (
    <div>
      <FeaturesSection />

      {/* Architecture diagram */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-10">
            System <span className="gradient-text">Architecture</span>
          </h2>
          <div className="glass-card p-8 font-mono text-xs sm:text-sm leading-relaxed text-[var(--color-text-secondary)] overflow-x-auto">
            <pre>{`                         [ AVAILABLE SENSORS ]
                       /           |           \\
             (optional)            |            (optional)
                Audio            Video             EEG
                  |                |             (Phase 2)
              [HuBERT]        [MediaPipe]
                  |                |
           Prosody/Spectral    Landmarks
              Embeddings      & AU Dynamics
                  \\                |
                   \\               |
              ┌──────────────────────────────────────┐
              │   Modality Availability Embeddings   │
              ├──────────────────────────────────────┤
              │   Stochastic Modality Dropout (p=.3) │
              ├──────────────────────────────────────┤
              │   Cross-Attention Bottleneck Fusion   │
              │         (32 tokens × 8 heads)        │
              └──────────────┬───────────────────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
    [ Classification Head ]     [ Confidence Head ]
    ├── Binary (D/C)            ├── MC Dropout (×20)
    ├── Severity (PHQ-8)        └── Calibrated Score
    └── Attention Maps

              ▼
    [ Clinical Interpretation ]
    └── Risk / Protective Factors`}</pre>
          </div>
        </div>
      </section>

      {/* Pipeline steps */}
      <section className="py-16 px-6">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Processing <span className="gradient-text">Pipeline</span>
          </h2>
          <div className="space-y-6">
            {[
              { step: "1", title: "Record or Upload", desc: "Capture audio, video, or both directly in your browser. Raw media is processed locally and never stored.", color: "#6366f1" },
              { step: "2", title: "Feature Extraction", desc: "Audio: 136-dim vector (MFCC, spectral, F0, prosody). Video: MediaPipe FaceMesh → AU dynamics, blink rate, head pose.", color: "#06b6d4" },
              { step: "3", title: "Cross-Modal Fusion", desc: "Attention bottleneck compresses multi-modal signals through 32 learnable tokens. Modality dropout ensures resilience to missing inputs.", color: "#10b981" },
              { step: "4", title: "Prediction + Confidence", desc: "Binary classification + severity regression with Monte Carlo Dropout for calibrated confidence estimation.", color: "#f59e0b" },
              { step: "5", title: "Clinical Interpretation", desc: "Risk and protective factors are extracted and mapped to DSM-5 symptom clusters for clinician-interpretable output.", color: "#ef4444" },
            ].map((item) => (
              <div key={item.step} className="flex gap-6 items-start">
                <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-white font-bold text-lg shrink-0"
                  style={{ background: `linear-gradient(135deg, ${item.color}, ${item.color}bb)` }}>
                  {item.step}
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-1">{item.title}</h3>
                  <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6 text-center">
        <Link href="/analyze"
          className="px-8 py-3.5 rounded-xl font-semibold text-white transition-all hover:scale-105 glow-primary inline-block"
          style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
          Try It Now →
        </Link>
      </section>
    </div>
  );
}
