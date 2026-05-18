export interface PredictionResult {
  is_model_trained: boolean;
  prediction: string | null;
  depression_probability: number | null;
  severity_level: string | null;
  severity_score: number | null;
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
