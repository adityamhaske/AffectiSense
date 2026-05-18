"use client";

export default function HeroSection() {
  return (
    <section className="relative pt-32 pb-16 px-6 overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute top-20 left-1/4 w-96 h-96 rounded-full opacity-20 blur-3xl pointer-events-none"
        style={{ background: "radial-gradient(circle, #6366f1, transparent 70%)" }} />
      <div className="absolute top-40 right-1/4 w-80 h-80 rounded-full opacity-15 blur-3xl pointer-events-none"
        style={{ background: "radial-gradient(circle, #06b6d4, transparent 70%)" }} />

      <div className="max-w-4xl mx-auto text-center relative z-10">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-elevated)] text-xs font-medium text-[var(--color-text-secondary)] mb-8 animate-fade-in-up">
          <span className="w-2 h-2 rounded-full bg-[var(--color-success)] animate-pulse" />
          Modality-Resilient AI • Audio + Video
        </div>

        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-tight tracking-tight mb-6 animate-fade-in-up"
          style={{ animationDelay: "0.1s" }}>
          Multimodal{" "}
          <span className="gradient-text">Mental Health</span>
          <br />
          Assessment Platform
        </h1>

        <p className="text-lg sm:text-xl text-[var(--color-text-secondary)] max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in-up"
          style={{ animationDelay: "0.2s" }}>
          Upload speech recordings or facial video for AI-powered depression screening
          with <span className="text-[var(--color-text-primary)] font-medium">calibrated confidence scores</span> and{" "}
          <span className="text-[var(--color-text-primary)] font-medium">clinically interpretable explanations</span>.
        </p>

        <div className="flex flex-wrap justify-center gap-4 mb-12 animate-fade-in-up"
          style={{ animationDelay: "0.3s" }}>
          <a href="#analyze"
            className="px-7 py-3 rounded-xl font-semibold text-white transition-all hover:scale-105 glow-primary"
            style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
            Start Analysis →
          </a>
          <a href="#features"
            className="px-7 py-3 rounded-xl font-semibold text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-primary)] transition-all hover:text-[var(--color-text-primary)]">
            How It Works
          </a>
        </div>

        {/* Stats bar */}
        <div className="flex flex-wrap justify-center gap-8 sm:gap-16 animate-fade-in-up"
          style={{ animationDelay: "0.4s" }}>
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
  );
}
