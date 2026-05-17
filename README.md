# AffectiSense — Multimodal Mental Health Assessment Platform

<p align="center">
  <img src="https://img.shields.io/badge/Acoustics-HuBERT%20%7C%20Whisper-blueviolet?style=for-the-badge" alt="Audio Badge" />
  <img src="https://img.shields.io/badge/Computer_Vision-MediaPipe%20%7C%20ViViT-deepskyblue?style=for-the-badge" alt="Video Badge" />
  <img src="https://img.shields.io/badge/Neurophysiology-EEGNet%20%7C%20GAT-emerald?style=for-the-badge" alt="EEG Badge" />
  <img src="https://img.shields.io/badge/AI_Engine-FastAPI%20%7C%20PyTorch-orange?style=for-the-badge" alt="Backend Badge" />
  <img src="https://img.shields.io/badge/Clinical_AI-DSM--5%20%7C%20RAG-crimson?style=for-the-badge" alt="Clinical AI Badge" />
</p>

AffectiSense is an intelligent, clinically oriented, and **modality-resilient** mental health assessment platform. By fusing neurophysiological (EEG), vocal biomarker (audio), and behavioral (facial expressions/video) signals, AffectiSense provides objective, highly interpretable depression screening with calibrated confidence metrics. 

Unlike traditional multi-modal frameworks that fail when certain inputs are missing, **AffectiSense is designed to gracefully degrade**—producing meaningful severity and diagnostic predictions whether given a single modality (e.g., audio only) or the full EAV triplet.

---

## 🌟 Core Pillars

1. **Modality Resilience by Design**  
   Uses an **Attention Bottleneck Fusion Core** trained with **Modality Dropout ($p=0.3$)**. The system supports any subset of available inputs (Audio, Video, EEG) dynamically, ensuring clinical viability in messy, real-world healthcare environments.
2. **Interpretability & Trust (XAI)**  
   Moves away from opaque black boxes. Multimodal attention weights are mathematically projected back to specific time-frequency ranges, vocal biomarkers, or facial action units (AUs), mapping them directly to **DSM-5 symptom clusters**.
3. **Calibrated Confidence**  
   Implements uncertainty estimation through Monte Carlo Dropout and quality-assessment checks. Clinicians are explicitly informed of *when* to trust or review the system's output.
4. **Privacy-by-Design (HIPAA Aware)**  
   Visual streams are processed locally via landmark detection—discarding raw video frames instantly and saving only lightweight, de-identified spatial coordinates to guarantee patient privacy.

---

## 🧠 Technical Architecture

```
                                 [ AVAILABLE SENSORS ]
                               /           |           \
                     (optional)            |            (optional)
                        EEG              Audio            Video
                         |                 |                |
                     [EEGNet]          [HuBERT]        [MediaPipe]
                         |                 |                |
                     GAT Graph       Prosody/Spectral   Landmarks
                     Features           Embeddings     & AU Dynamics
                         \                 |               /
                          \                |              /
                     ┌──────────────────────────────────────────┐
                     │   Modality Availability Embeddings       │
                     ├──────────────────────────────────────────┤
                     │   Stochastic Modality Dropout Core       │
                     ├──────────────────────────────────────────┤
                     │   Cross-Attention Bottleneck Fusion      │
                     └────────────────────┬─────────────────────┘
                                          |
                        ┌─────────────────┴─────────────────┐
                        │                                   │
             [ Clinical Predictor ]               [ Confidence Head ]
             ├── Binary Classification            ├── Epistemic Uncertainty
             ├── PHQ-8 Severity Score             └── Input Quality Calibrator
             └── Attention Map Projection
                        │
                        ▼
             [ DSM-5 Grounded Narrative ]
             └── RAG Agent Clinical Co-Pilot
```

### 1. Modality Encoders
* **Audio (Vocal Biomarkers):** Custom projection head running over frozen state-of-the-art self-supervised foundation embeddings (**HuBERT-Base** layers 6–9) to capture speech rate, tone, rhythm, and acoustic biomarkers of depression.
* **Video (Facial Expression Dynamics):** **Temporal Vision Transformer** running over localized **MediaPipe FaceMesh** coordinates (468 landmarks + 52 FACS blendshapes) to trace hyper-specific facial action unit (AU) dynamics, gaze patterns, and blink behaviors.
* **EEG (Neurophysiology - *Phase 2*):** Hybrid **EEGNet + Graph Attention Network (GAT)** mapping functional connectivity across 128-channel or wearable 3-electrode systems.

### 2. Attention Bottleneck Fusion
Instead of high-dimensional direct concatenation, input embeddings are compressed through a small set of latent bottleneck tokens. This forces the network to distill complementary modal signals and protects against noisy or corrupted data.

---

## 🗂 Data & Benchmark Strategy

AffectiSense is developed and validated using leading clinical and behavioral datasets:
* **DAIC-WOZ / AVEC:** Multimodal clinical interview corpus containing raw audio, video features, and transcriptions mapped to PHQ-8 scores.
* **MODMA:** Multi-modal Open Dataset for Mental-disorder Analysis containing high-density 128-channel EEG alongside audio signals.
* **D-Vlog:** Large-scale spontaneous video logs labeled for depression classification.
* **CMU-MOSEI:** Emotional and behavioral speech expressions in complex natural environments.

---

## 🛠 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/            # API routing and gateways
│   │   ├── core/           # Configs, secrets, and system state
│   │   ├── models/         # PyTorch multimodal layers & fusion core
│   │   ├── pipelines/      # Modality feature processing & orchestration
│   │   ├── schemas/        # Request/Response Pydantic models
│   │   └── utils/          # Signal processing helpers
│   ├── requirements.txt    # CPU-optimized ML & web stack
│   └── Dockerfile
├── frontend/               # Next.js 14 Premium Clinical UI (Next Steps)
├── configs/                # Hyperparameters, pipelines & thresholds
├── data/
│   ├── raw/                # Protected raw local files
│   └── models/             # Locally cached model checkpoints
└── README.md
```

---

## 🚀 Execution Roadmap

### 🏁 Phase 1: MVP (Days 1–30) — *Current Phase*
* [x] Restructured repository to clean monorepo architecture.
* [x] Formulated pydantic validation, pipeline schemas, and config files.
* [ ] Implement full local CPU-optimized feature-extraction pipelines (Librosa + MediaPipe).
* [ ] Design attention bottleneck fusion and modality dropout model.
* [ ] Deploy FastAPI engine with streaming-ready upload handlers.
* [ ] Create Next.js 14 clinical dashboard with diagnostic gauges and attention heatmap overlays.

### 🔬 Phase 2: SOTA Deep Learning (Days 31–60)
* [ ] Integrate frozen pre-trained foundation models (HuBERT/Whisper) for audio features.
* [ ] Implement sequence-to-sequence temporal video transformer.
* [ ] Evaluate all 7 modality combinations (A, V, E, AV, AE, VE, AVE) on standard open datasets.
* [ ] Add EEGNet graph-network support for neurophysiological inputs.

### 💼 Phase 3: Clinical Platform (Days 61–90)
* [ ] Integrate RAG-grounded LLM clinical co-pilot for automated diagnostic draft reports.
* [ ] Complete end-to-end HIPAA security hardening (Audit logging, RBAC, KMS-encryption).
* [ ] Dockerize production deployments via Kubernetes & Triton serving.

---

## 🧑‍💻 Quickstart

### Prerequisites
* Python 3.10+
* FFmpeg (for audio/video encoding)

### Installation
1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/adityamhaske/AffectiSense.git
   cd AffectiSense
   ```

2. Setup virtual environment and install backend dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. Run local backend server:
   ```bash
   uvicorn backend.app.api.main:app --reload
   ```

---

## 🤝 License & Disclaimer
This platform is intended strictly for educational, portfolio, and academic research purposes. It is **not** a diagnostic medical device. It should only be evaluated as an auxiliary clinical support tool.

Licensed under the MIT License.
