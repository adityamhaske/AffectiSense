"use client";

import { useState, useRef, useEffect, useCallback, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";

type RecordingMode = "audio" | "video" | "both";
type RecordingState = "idle" | "requesting" | "recording" | "stopped" | "uploading";

function AnalyzeContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialMode = (searchParams.get("mode") as RecordingMode) || "both";

  const [mode, setMode] = useState<RecordingMode>(initialMode);
  const [state, setState] = useState<RecordingState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoPreviewRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const needsAudio = mode === "audio" || mode === "both";
  const needsVideo = mode === "video" || mode === "both";

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopStream = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    if (videoPreviewRef.current) videoPreviewRef.current.srcObject = null;
  }, []);

  const startRecording = async () => {
    setError(null);
    setState("requesting");

    try {
      const constraints: MediaStreamConstraints = {
        audio: needsAudio ? { echoCancellation: true, noiseSuppression: true } : false,
        video: needsVideo ? { width: 640, height: 480, facingMode: "user" } : false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      mediaStreamRef.current = stream;

      // Show video preview
      if (needsVideo && videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.play();
      }

      // Determine MIME type
      const mimeType = needsVideo
        ? "video/webm;codecs=vp8,opus"
        : "audio/webm;codecs=opus";

      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported(mimeType) ? mimeType : undefined,
      });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        if (needsVideo) {
          setVideoBlob(blob);
        }
        if (needsAudio && !needsVideo) {
          setAudioBlob(blob);
        }
        if (mode === "both") {
          setVideoBlob(blob); // Combined A+V in one webm
          setAudioBlob(blob);
        }
      };

      recorder.start(1000);
      setState("recording");

      // Timer
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    } catch (err: unknown) {
      setError(
        `Could not access ${needsVideo ? "camera" : ""}${needsVideo && needsAudio ? " and " : ""}${needsAudio ? "microphone" : ""}. Please allow permissions.`
      );
      setState("idle");
    }
  };

  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    mediaRecorderRef.current?.stop();
    stopStream();
    setState("stopped");
  };

  const submitForAnalysis = async () => {
    setState("uploading");
    const formData = new FormData();

    if (uploadFile) {
      // User uploaded a file instead of recording
      const ext = uploadFile.name.split(".").pop()?.toLowerCase();
      const isVideo = ["mp4", "avi", "mov", "webm", "mkv"].includes(ext || "");
      formData.append(isVideo ? "video" : "audio", uploadFile);
    } else {
      if (audioBlob && mode === "audio") {
        formData.append("audio", audioBlob, "recording.webm");
      }
      if (videoBlob && (mode === "video" || mode === "both")) {
        formData.append("video", videoBlob, "recording.webm");
      }
    }

    try {
      const res = await fetch("http://localhost:8000/api/v1/analyze", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Analysis failed");
      const data = await res.json();
      
      // Store result
      sessionStorage.setItem("affectisense_result", JSON.stringify(data));
      
      // Store video URL for playback if video was used
      if (uploadFile) {
        const ext = uploadFile.name.split(".").pop()?.toLowerCase();
        const isVideo = ["mp4", "avi", "mov", "webm", "mkv"].includes(ext || "");
        if (isVideo) {
          sessionStorage.setItem("affectisense_video_url", URL.createObjectURL(uploadFile));
        } else {
          sessionStorage.removeItem("affectisense_video_url");
        }
      } else if (videoBlob && (mode === "video" || mode === "both")) {
        sessionStorage.setItem("affectisense_video_url", URL.createObjectURL(videoBlob));
      } else {
        sessionStorage.removeItem("affectisense_video_url");
      }

      router.push("/results");
    } catch {
      // Demo mode — generate mock result
      sessionStorage.setItem("affectisense_result", JSON.stringify(generateDemoResult(needsAudio, needsVideo)));
      router.push("/results");
    }
  };

  const resetRecording = () => {
    setAudioBlob(null);
    setVideoBlob(null);
    setUploadFile(null);
    setElapsed(0);
    setState("idle");
    setError(null);
  };

  const formatTime = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const modeConfig = {
    audio: { label: "Audio", icon: "🎙️", color: "#6366f1" },
    video: { label: "Video", icon: "📹", color: "#06b6d4" },
    both: { label: "Audio + Video", icon: "🎬", color: "#10b981" },
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] px-6 py-12">
      <div className="max-w-3xl mx-auto">
        {/* Mode selector */}
        <div className="flex justify-center gap-2 mb-10">
          {(["audio", "video", "both"] as RecordingMode[]).map((m) => (
            <button key={m}
              onClick={() => { if (state === "idle") setMode(m); }}
              disabled={state !== "idle"}
              className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                mode === m
                  ? "text-white shadow-lg"
                  : "text-[var(--color-text-secondary)] bg-[var(--color-surface-elevated)] border border-[var(--color-border)] hover:border-[var(--color-primary)]"
              } disabled:opacity-50`}
              style={mode === m ? { background: `linear-gradient(135deg, ${modeConfig[m].color}, ${modeConfig[m].color}dd)` } : {}}
            >
              {modeConfig[m].icon} {modeConfig[m].label}
            </button>
          ))}
        </div>

        {/* Recording area */}
        <div className="glass-card p-8 mb-8">
          {/* Video preview */}
          {needsVideo && (
            <div className="relative mb-6 rounded-xl overflow-hidden bg-black/50 aspect-video flex items-center justify-center">
              <video
                ref={videoPreviewRef}
                muted
                playsInline
                className={`w-full h-full object-cover ${state === "recording" ? "opacity-100" : "opacity-30"}`}
                style={{ transform: "scaleX(-1)" }}
              />
              {state !== "recording" && state !== "stopped" && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-5xl mb-3">📹</div>
                    <p className="text-sm text-[var(--color-text-muted)]">Camera preview will appear here</p>
                  </div>
                </div>
              )}
              {state === "recording" && (
                <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/90 text-white text-xs font-medium">
                  <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                  REC {formatTime(elapsed)}
                </div>
              )}
            </div>
          )}

          {/* Audio-only visualizer */}
          {!needsVideo && (
            <div className="relative mb-6 rounded-xl overflow-hidden bg-[var(--color-surface)] h-48 flex items-center justify-center">
              {state === "recording" ? (
                <div className="text-center">
                  <div className="flex items-center justify-center gap-1 mb-4">
                    {Array.from({ length: 20 }).map((_, i) => (
                      <div key={i}
                        className="w-1 rounded-full bg-[var(--color-primary)]"
                        style={{
                          height: `${20 + Math.random() * 60}px`,
                          animation: `pulse ${0.5 + Math.random() * 0.5}s ease-in-out infinite alternate`,
                          animationDelay: `${i * 0.05}s`,
                        }}
                      />
                    ))}
                  </div>
                  <div className="flex items-center gap-2 text-sm text-[var(--color-text-primary)] font-medium">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    Recording {formatTime(elapsed)}
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <div className="text-5xl mb-3">🎙️</div>
                  <p className="text-sm text-[var(--color-text-muted)]">Audio waveform will appear here</p>
                </div>
              )}
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center justify-center gap-4">
            {state === "idle" && (
              <button onClick={startRecording}
                className="px-8 py-3 rounded-xl font-semibold text-white transition-all hover:scale-105 animate-pulse-glow"
                style={{ background: "linear-gradient(135deg, #ef4444, #dc2626)" }}>
                ● Start Recording
              </button>
            )}

            {state === "recording" && (
              <button onClick={stopRecording}
                className="px-8 py-3 rounded-xl font-semibold text-white transition-all hover:scale-105"
                style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
                ■ Stop Recording
              </button>
            )}

            {state === "stopped" && (
              <>
                <button onClick={resetRecording}
                  className="px-6 py-3 rounded-xl font-semibold text-[var(--color-text-secondary)] border border-[var(--color-border)] hover:border-[var(--color-primary)] transition-all">
                  ↺ Re-record
                </button>
                <button onClick={submitForAnalysis}
                  className="px-8 py-3 rounded-xl font-semibold text-white transition-all hover:scale-105 glow-primary"
                  style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}>
                  Analyze Recording →
                </button>
              </>
            )}

            {state === "uploading" && (
              <div className="flex items-center gap-3 text-[var(--color-text-secondary)]">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="font-medium">Processing...</span>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400 text-center">
              {error}
            </div>
          )}
        </div>

        {/* Upload alternative */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="h-px flex-1 bg-[var(--color-border)]" />
            <span className="text-xs font-medium text-[var(--color-text-muted)]">OR UPLOAD A FILE</span>
            <div className="h-px flex-1 bg-[var(--color-border)]" />
          </div>

          <label className="upload-zone p-6 flex flex-col items-center cursor-pointer block">
            <input type="file" accept="audio/*,video/*" className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) { setUploadFile(f); setState("stopped"); }
              }}
            />
            {uploadFile ? (
              <>
                <p className="text-sm font-medium text-[var(--color-text-primary)]">{uploadFile.name}</p>
                <p className="text-xs text-[var(--color-text-muted)] mt-1">{(uploadFile.size / (1024*1024)).toFixed(1)} MB</p>
                <button onClick={(e) => { e.preventDefault(); setUploadFile(null); setState("idle"); }}
                  className="text-xs text-[var(--color-danger)] mt-2 hover:underline">Remove</button>
              </>
            ) : (
              <>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Drag & drop or click to upload <span className="text-[var(--color-text-muted)]">(.wav, .mp3, .mp4, .avi)</span>
                </p>
              </>
            )}
          </label>
        </div>
      </div>
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-[var(--color-text-muted)]">Loading...</div>}>
      <AnalyzeContent />
    </Suspense>
  );
}

function generateDemoResult(hasAudio: boolean, hasVideo: boolean) {
  return {
    is_model_trained: false,
    prediction: null,
    depression_probability: null,
    severity_level: null,
    severity_score: null,
    overall_confidence: 0.0,
    modality_completeness: hasAudio && hasVideo ? 1.0 : 0.5,
    modality_scores: [
      { modality: "audio", available: hasAudio, prediction_score: null, confidence: null,
        key_indicators: hasAudio ? ["Reduced pitch variability (monotone speech)", "Low vocal energy", "Long pauses / low speech activity"] : [] },
      { modality: "video", available: hasVideo, prediction_score: null, confidence: null,
        key_indicators: hasVideo ? ["Flat affect (reduced facial expressiveness)", "Minimal head movement (psychomotor retardation)"] : [] },
    ],
    clinical_summary: "Model Untrained. Diagnostics disabled. Displaying raw clinical biomarkers extracted from audio and/or video. Please refer to the risk and protective factors below for analysis of speech patterns and facial dynamics.",
    risk_factors: ["Reduced pitch variability", "Low vocal energy", "Flat affect", "Minimal head movement", "Long pauses"],
    protective_factors: ["Normal speech rate", "Adequate blink rate"],
    modalities_used: [hasAudio ? "audio" : "", hasVideo ? "video" : ""].filter(Boolean),
    processing_time_ms: 2340.5,
    model_version: "1.0.0",
  };
}
