"use client";

import { useState, useRef, useCallback } from "react";

interface UploadSectionProps {
  onAnalyze: (audio: File | null, video: File | null) => void;
  isAnalyzing: boolean;
}

export default function UploadSection({ onAnalyze, isAnalyzing }: UploadSectionProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [audioDragOver, setAudioDragOver] = useState(false);
  const [videoDragOver, setVideoDragOver] = useState(false);
  const audioRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent, type: "audio" | "video") => {
    e.preventDefault();
    type === "audio" ? setAudioDragOver(false) : setVideoDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      type === "audio" ? setAudioFile(file) : setVideoFile(file);
    }
  }, []);

  const handleSubmit = () => {
    if (!audioFile && !videoFile) return;
    onAnalyze(audioFile, videoFile);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <section id="analyze" className="py-20 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Upload Your <span className="gradient-text">Data</span>
          </h2>
          <p className="text-[var(--color-text-secondary)] max-w-xl mx-auto">
            Provide audio and/or video for analysis. The system works with any combination — upload what you have.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* Audio upload */}
          <div
            className={`upload-zone p-8 text-center ${audioDragOver ? "drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setAudioDragOver(true); }}
            onDragLeave={() => setAudioDragOver(false)}
            onDrop={(e) => handleDrop(e, "audio")}
            onClick={() => audioRef.current?.click()}
          >
            <input ref={audioRef} type="file" accept="audio/*" className="hidden"
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)} />

            <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center"
              style={{ background: audioFile ? "rgba(16, 185, 129, 0.15)" : "rgba(99, 102, 241, 0.1)" }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
                stroke={audioFile ? "#10b981" : "#6366f1"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </div>

            <h3 className="text-lg font-semibold mb-1">
              {audioFile ? audioFile.name : "Audio Recording"}
            </h3>
            <p className="text-sm text-[var(--color-text-muted)] mb-3">
              {audioFile
                ? formatSize(audioFile.size)
                : "Drag & drop or click to upload • .wav, .mp3, .flac"}
            </p>
            {audioFile && (
              <button onClick={(e) => { e.stopPropagation(); setAudioFile(null); }}
                className="text-xs text-[var(--color-danger)] hover:underline">
                Remove
              </button>
            )}
          </div>

          {/* Video upload */}
          <div
            className={`upload-zone p-8 text-center ${videoDragOver ? "drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setVideoDragOver(true); }}
            onDragLeave={() => setVideoDragOver(false)}
            onDrop={(e) => handleDrop(e, "video")}
            onClick={() => videoRef.current?.click()}
          >
            <input ref={videoRef} type="file" accept="video/*" className="hidden"
              onChange={(e) => setVideoFile(e.target.files?.[0] || null)} />

            <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center"
              style={{ background: videoFile ? "rgba(16, 185, 129, 0.15)" : "rgba(6, 182, 212, 0.1)" }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
                stroke={videoFile ? "#10b981" : "#06b6d4"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
              </svg>
            </div>

            <h3 className="text-lg font-semibold mb-1">
              {videoFile ? videoFile.name : "Facial Video"}
            </h3>
            <p className="text-sm text-[var(--color-text-muted)] mb-3">
              {videoFile
                ? formatSize(videoFile.size)
                : "Drag & drop or click to upload • .mp4, .avi, .mov"}
            </p>
            {videoFile && (
              <button onClick={(e) => { e.stopPropagation(); setVideoFile(null); }}
                className="text-xs text-[var(--color-danger)] hover:underline">
                Remove
              </button>
            )}
          </div>
        </div>

        {/* Modality status bar */}
        <div className="glass-card p-4 flex items-center justify-between flex-wrap gap-4 mb-6">
          <div className="flex items-center gap-4">
            <span className="text-sm text-[var(--color-text-secondary)]">Modalities:</span>
            <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full ${
              audioFile ? "bg-[rgba(16,185,129,0.15)] text-[var(--color-success)]" : "bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${audioFile ? "bg-[var(--color-success)]" : "bg-[var(--color-text-muted)]"}`} />
              Audio {audioFile ? "✓" : "—"}
            </span>
            <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full ${
              videoFile ? "bg-[rgba(16,185,129,0.15)] text-[var(--color-success)]" : "bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${videoFile ? "bg-[var(--color-success)]" : "bg-[var(--color-text-muted)]"}`} />
              Video {videoFile ? "✓" : "—"}
            </span>
          </div>
          <button
            onClick={handleSubmit}
            disabled={(!audioFile && !videoFile) || isAnalyzing}
            className="px-6 py-2.5 rounded-xl font-semibold text-sm text-white transition-all hover:scale-105 disabled:opacity-40 disabled:hover:scale-100 disabled:cursor-not-allowed"
            style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}
          >
            {isAnalyzing ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing...
              </span>
            ) : "Run Analysis"}
          </button>
        </div>
      </div>
    </section>
  );
}
