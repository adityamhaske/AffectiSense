import Link from "next/link";

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute top-20 left-1/4 w-96 h-96 rounded-full opacity-20 blur-3xl pointer-events-none"
        style={{ background: "radial-gradient(circle, #6366f1, transparent 70%)" }} />
      <div className="absolute top-40 right-1/4 w-80 h-80 rounded-full opacity-15 blur-3xl pointer-events-none"
        style={{ background: "radial-gradient(circle, #06b6d4, transparent 70%)" }} />

      {/* Hero */}
      <section className="relative z-10 pt-20 pb-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-elevated)] text-xs font-medium text-[var(--color-text-secondary)] mb-8 animate-fade-in-up">
            <span className="w-2 h-2 rounded-full bg-[var(--color-success)] animate-pulse" />
            Modality-Resilient AI • Audio + Video
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-tight tracking-tight mb-6 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
            Multimodal{" "}
            <span className="gradient-text">Mental Health</span>
            <br />
            Assessment Platform
          </h1>

          <p className="text-lg sm:text-xl text-[var(--color-text-secondary)] max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
            Record speech or facial video directly in your browser for AI-powered depression screening with{" "}
            <span className="text-[var(--color-text-primary)] font-medium">calibrated confidence scores</span> and{" "}
            <span className="text-[var(--color-text-primary)] font-medium">clinically interpretable explanations</span>.
          </p>

          <div className="flex flex-wrap justify-center gap-4 mb-16 animate-fade-in-up" style={{ animationDelay: "0.3s" }}>
            <Link href="/analyze"
              className="px-8 py-3.5 rounded-xl font-semibold text-white transition-all hover:scale-105 glow-primary"
              style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
              Upload Analysis →
            </Link>
            <Link href="/interview"
              className="px-8 py-3.5 rounded-xl font-semibold text-white transition-all hover:scale-105"
              style={{ background: "linear-gradient(135deg, #06b6d4, #0891b2)" }}>
              Start Conversational Interview
            </Link>
            <Link href="/about"
              className="px-8 py-3.5 rounded-xl font-semibold text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-primary)] transition-all hover:text-[var(--color-text-primary)]">
              How It Works
            </Link>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap justify-center gap-8 sm:gap-16 animate-fade-in-up" style={{ animationDelay: "0.4s" }}>
            {[
              { value: "2", label: "Modalities" },
              { value: "136+", label: "Audio Features" },
              { value: "468", label: "Facial Landmarks" },
              { value: "<3s", label: "Inference" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-2xl font-bold gradient-text">{stat.value}</div>
                <div className="text-xs text-[var(--color-text-muted)] mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Mode cards */}
      <section className="relative z-10 pb-24 px-6">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-bold text-center mb-10">
            Choose Your <span className="gradient-text">Recording Mode</span>
          </h2>
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { icon: "🎙️", title: "Audio Only", desc: "Record speech for vocal biomarker analysis — pitch, prosody, energy, and speech patterns.", href: "/analyze?mode=audio", color: "#6366f1" },
              { icon: "📹", title: "Video Only", desc: "Record facial video for expression dynamics — action units, blink rate, head pose.", href: "/analyze?mode=video", color: "#06b6d4" },
              { icon: "🎬", title: "Audio + Video", desc: "Record both simultaneously for maximum accuracy with full cross-modal fusion.", href: "/analyze?mode=both", color: "#10b981" },
            ].map((mode) => (
              <Link key={mode.title} href={mode.href}
                className="glass-card p-8 text-center group hover:scale-[1.02] transition-all">
                <div className="text-4xl mb-4">{mode.icon}</div>
                <h3 className="text-lg font-semibold mb-2 group-hover:text-[var(--color-primary)] transition-colors">{mode.title}</h3>
                <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">{mode.desc}</p>
                <div className="mt-4 text-xs font-medium transition-colors" style={{ color: mode.color }}>
                  Start →
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[var(--color-border)] py-8 px-6 text-center">
        <p className="text-xs text-[var(--color-text-muted)] max-w-md mx-auto">
          For educational and research purposes only. Not a diagnostic medical device. Clinical follow-up is always recommended.
        </p>
      </footer>
    </div>
  );
}
