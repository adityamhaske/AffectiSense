"use client";

export default function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] py-10 px-6">
      <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-muted)]">
          <span className="gradient-text font-semibold">AffectiSense</span>
          <span>•</span>
          <span>Multimodal Mental Health Assessment</span>
        </div>
        <div className="text-xs text-[var(--color-text-muted)] text-center sm:text-right max-w-md">
          For educational and research purposes only. Not a diagnostic medical device.
          Clinical follow-up is always recommended.
        </div>
      </div>
    </footer>
  );
}
