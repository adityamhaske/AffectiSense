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
