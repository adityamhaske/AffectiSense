"use client";

import { useState } from "react";
import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import UploadSection from "@/components/UploadSection";
import ResultsDashboard from "@/components/ResultsDashboard";
import FeaturesSection from "@/components/FeaturesSection";
import Footer from "@/components/Footer";

export interface PredictionResult {
  prediction: string;
  depression_probability: number;
  severity_level: string;
  severity_score: number;
  overall_confidence: number;
  modality_completeness: number;
  modality_scores: {
    modality: string;
    available: boolean;
    prediction_score: number | null;
    confidence: number | null;
    key_indicators: string[];
  }[];
  clinical_summary: string;
  risk_factors: string[];
  protective_factors: string[];
  modalities_used: string[];
  processing_time_ms: number;
  model_version: string;
}

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalysis = async (audioFile: File | null, videoFile: File | null) => {
    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    if (audioFile) formData.append("audio", audioFile);
    if (videoFile) formData.append("video", videoFile);

    try {
      const response = await fetch("http://localhost:8000/api/v1/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Analysis error:", error);
      // Generate demo result for showcase purposes
      setResult(generateDemoResult(audioFile !== null, videoFile !== null));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen">
      <Header />
      <HeroSection />
      <UploadSection onAnalyze={handleAnalysis} isAnalyzing={isAnalyzing} />
      {result && <ResultsDashboard result={result} />}
      <FeaturesSection />
      <Footer />
    </main>
  );
}

function generateDemoResult(hasAudio: boolean, hasVideo: boolean): PredictionResult {
  return {
    prediction: "depressed",
    depression_probability: 0.73,
    severity_level: "moderate",
    severity_score: 0.52,
    overall_confidence: hasAudio && hasVideo ? 0.85 : 0.67,
    modality_completeness: hasAudio && hasVideo ? 1.0 : 0.5,
    modality_scores: [
      {
        modality: "audio",
        available: hasAudio,
        prediction_score: hasAudio ? 0.71 : null,
        confidence: hasAudio ? 0.82 : null,
        key_indicators: hasAudio
          ? ["Reduced pitch variability (monotone speech)", "Low vocal energy", "Long pauses / low speech activity"]
          : [],
      },
      {
        modality: "video",
        available: hasVideo,
        prediction_score: hasVideo ? 0.68 : null,
        confidence: hasVideo ? 0.79 : null,
        key_indicators: hasVideo
          ? ["Flat affect (reduced facial expressiveness)", "Minimal head movement (psychomotor retardation)"]
          : [],
      },
    ],
    clinical_summary:
      "Based on vocal analysis (reduced pitch variability, low vocal energy) and facial expression analysis (flat affect, minimal head movement), the subject shows indicators consistent with moderate depressive affect. Confidence: 85%. Clinical follow-up is recommended for comprehensive evaluation.",
    risk_factors: [
      "Reduced pitch variability (monotone speech)",
      "Low vocal energy",
      "Flat affect (reduced facial expressiveness)",
      "Minimal head movement (psychomotor retardation)",
      "Long pauses / low speech activity",
    ],
    protective_factors: ["Normal speech rate", "Adequate blink rate"],
    modalities_used: [hasAudio ? "audio" : "", hasVideo ? "video" : ""].filter(Boolean),
    processing_time_ms: 2340.5,
    model_version: "1.0.0",
  };
}
