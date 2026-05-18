"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";

interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface AnalysisResult {
  transcript: string;
  sentiment_label: string;
  sentiment_score: number;
  clinical_themes: Record<string, number>;
  next_question: string;
}

export default function InterviewPage() {
  const router = useRouter();
  
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Hi there. I'm here to listen. How have you been feeling over the past week?" }
  ]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      mediaRecorderRef.current?.stream.getTracks().forEach(t => t.stop());
    };
  }, []);

  const formatTime = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
        await processAudioSegment(audioBlob);
      };

      recorder.start(1000);
      setIsRecording(true);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed(t => t + 1), 1000);
    } catch (err) {
      console.error("Microphone access denied", err);
      alert("Microphone access is required for the interview.");
    }
  };

  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current?.stream.getTracks().forEach(t => t.stop());
    setIsRecording(false);
    setIsProcessing(true);
  };

  const processAudioSegment = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "response.webm");
    
    // We only send the last few messages for context
    const recentHistory = messages.filter(m => m.role !== "system").slice(-4);
    formData.append("history", JSON.stringify(recentHistory));

    try {
      const res = await fetch("http://localhost:8000/api/v1/interview/process", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Processing failed");

      const data: AnalysisResult = await res.json();
      
      setMessages(prev => [
        ...prev,
        { role: "user", content: data.transcript || "[Inaudible response]" },
        { role: "assistant", content: data.next_question }
      ]);
      
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { role: "user", content: "[Audio response captured]" },
        { role: "assistant", content: "I'm having trouble processing that right now. Could you repeat it, or tell me more?" }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const endInterview = () => {
    // In a real app, we would send the full history to calculate a final score
    router.push("/");
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex flex-col max-w-4xl mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Clinical Assistant Interview</h1>
        <button onClick={endInterview} className="px-4 py-2 rounded-lg text-sm font-medium border border-[var(--color-border)] hover:border-[var(--color-danger)] hover:text-[var(--color-danger)] transition-colors">
          End Interview
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 glass-card p-6 mb-6 overflow-y-auto flex flex-col gap-6" style={{ maxHeight: "60vh" }}>
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[80%] p-4 rounded-2xl ${
              msg.role === "user" 
                ? "bg-[var(--color-primary)] text-white rounded-br-none" 
                : "bg-[var(--color-surface-elevated)] border border-[var(--color-border)] rounded-bl-none"
            }`}>
              {msg.role === "assistant" && <span className="block text-xs font-bold mb-1 opacity-70">AI Clinical Assistant</span>}
              <p className="leading-relaxed">{msg.content}</p>
            </div>
          </div>
        ))}
        {isProcessing && (
          <div className="flex justify-start">
             <div className="bg-[var(--color-surface-elevated)] border border-[var(--color-border)] p-4 rounded-2xl rounded-bl-none flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[var(--color-text-muted)] animate-bounce" style={{ animationDelay: "0ms" }}></div>
                <div className="w-2 h-2 rounded-full bg-[var(--color-text-muted)] animate-bounce" style={{ animationDelay: "150ms" }}></div>
                <div className="w-2 h-2 rounded-full bg-[var(--color-text-muted)] animate-bounce" style={{ animationDelay: "300ms" }}></div>
             </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Recording Controls */}
      <div className="glass-card p-6 flex flex-col items-center justify-center">
        {!isRecording && !isProcessing ? (
          <button 
            onClick={startRecording}
            className="w-20 h-20 rounded-full flex items-center justify-center glow-primary transition-transform hover:scale-105"
            style={{ background: "linear-gradient(135deg, #6366f1, #4f46e5)" }}
          >
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            </svg>
          </button>
        ) : isRecording ? (
          <div className="flex flex-col items-center">
            <button 
              onClick={stopRecording}
              className="w-20 h-20 rounded-full flex items-center justify-center animate-pulse-glow transition-transform hover:scale-105"
              style={{ background: "linear-gradient(135deg, #ef4444, #dc2626)" }}
            >
              <div className="w-8 h-8 bg-white rounded-sm"></div>
            </button>
            <div className="mt-4 font-mono text-[var(--color-danger)] font-bold">
              {formatTime(elapsed)}
            </div>
          </div>
        ) : (
          <div className="h-20 flex items-center justify-center text-[var(--color-text-muted)]">
            Analyzing response...
          </div>
        )}
        <p className="text-sm text-[var(--color-text-muted)] mt-4">
          {isRecording ? "Tap to stop responding" : "Tap to speak your response"}
        </p>
      </div>
    </div>
  );
}
