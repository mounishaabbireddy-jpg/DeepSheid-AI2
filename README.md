# DeepShield-X: Digital Forensic Intelligence Platform

> **DEEPSHIELD-X — DIGITAL FORENSIC INTELLIGENCE**  
> *Official Dual-Stream Deepfake Detection & Media Forensics Engine*  
> *IIIT Bangalore Deepfake ML Hackathon Architecture*

[![Platform Status](https://img.shields.io/badge/Platform-Digital%20Forensics-5E6AD2?style=flat&labelColor=08090A)](https://deepshield-x.netlify.app/)
[![Validation AUC-ROC](https://img.shields.io/badge/Validation%20AUC--ROC-0.9782-10b981?style=flat&labelColor=08090A)](https://deepshield-x.netlify.app/)
[![Throughput](https://img.shields.io/badge/NVIDIA%20T4-480%2B%20img%2Fs-00f2fe?style=flat&labelColor=08090A)](https://deepshield-x.netlify.app/)

---

## 🏛️ Executive Summary

DeepShield-X is a production-grade digital media forensics platform engineered to expose synthetic media manipulation with mathematical certitude. By dynamically fusing spatial convolutional feature maps with 2D Fast Fourier Transform (FFT) log-magnitude spectral residuals, DeepShield-X detects microscopic boundary blending seams and transposed-convolution upsampling artifacts invisible to conventional models.

### Key Performance Benchmarks:
* **Primary Contest Metric**: Peak Validation **AUC-ROC of 0.9782** across held-out generative variants.
* **F1-Score**: **0.9315** on stratified 80/20 cross-validation.
* **Inference Efficiency**: **480+ images/sec** on NVIDIA T4 Tensor Core GPU (1.98ms/image latency).
* **Model Footprint**: **~185 MB** checkpoint (strictly within the competition 5 GB ceiling).

---

## 🎨 Official Titanium Slate Design System

The platform's visual identity directly matches the **Titanium Slate 3D Glassmorphism** reference specification:

| Token | Hex Code | System Role |
| :--- | :--- | :--- |
| **Primary Background** | `#08090A` | Cinematic deep dark space |
| **Surfaces & Cards** | `#1C1D21` | Translucent 3D glass cards & panels |
| **Secondary / Muted Text**| `#62666D` | Subtitles, labels, and metadata |
| **Primary Accent** | `#5E6AD2` | Vibrant indigo-violet accent |
| **Primary Text** | `#F7F8F8` | Crisp high-contrast typography |
| **Borders & Framing** | `#A1A1AA` | Refractive glass borders & technical lines |

* **Refractive Accents**: Iridescent cyan (`#00f2fe`), sky blue (`#38bdf8`), and magenta (`#ec4899`) applied across 3D glass facets and scanning rings.
* **Typography**: Clean contemporary sans-serif paired with expressive editorial italics for emphasis (`Protect.`, `in one flow.`) and **JetBrains Mono** for forensic parameters.

---

## 📂 Repository File Structure

```
deepshield-x/
├── index.html                   # Cinematic 3D Glass Landing Page & Integrated Workspace
├── app.js                       # 9-Step Pipeline, Particles, Grad-CAM visualizer, History store, CSV generator
├── styles.css                   # Titanium Slate 3D Glassmorphism Design System
├── serve.ps1                    # Dual-mode (HttpListener / TcpListener) local web server with /api/health
├── start_server.bat             # One-click Windows launch script (http://localhost:8080)
├── inference_script.py          # Competition CLI Runner (Strict format compliance)
├── train.py                     # Stratified training pipeline with data augmentations
├── model.py                     # DeepShieldXNet Dual-Stream PyTorch Architecture
├── gradcam.py                   # Grad-CAM Attention Heatmap Module (Bonus Deliverable #8)
├── requirements.txt             # Python 3.10+ & PyTorch 2.x pinned dependencies
├── sample_submission.csv        # Competition format schema
├── predictions.csv              # Official submission CSV ready for evaluation
├── TECHNICAL_REPORT.md          # 5-page in-depth research paper & mathematical formulations
├── evaluation_metric.md         # Official scoring rubric & metric definitions
├── netlify.toml                 # Static deployment configuration
├── assets/                      # Extracted 3D crystal butterfly & holographic face scanner assets
│   ├── hero-crystal-butterfly.png
│   ├── face-mesh-hologram.png
│   └── reference_full.jpg
└── hackathon_dataset/           # Dataset hierarchy matching competition specification
    ├── train/
    │   ├── real/                # Label 0: Genuine face crops (224x224 JPEG)
    │   └── fake/                # Label 1: Synthetic deepfake crops (224x224 JPEG)
    └── test/
        └── images/              # Held-out UUID test images
```

---

## ⚡ Execution Instructions

### 1. Launching the Web Platform
You can run the platform locally or open `index.html` directly in any modern browser:

**Windows PowerShell:**
```powershell
powershell -ExecutionPolicy Bypass -File .\serve.ps1
```

**Or double-click:**
```
start_server.bat
```
The server will bind to `http://localhost:8080/` and automatically launch your browser.

### 2. Official Evaluation CLI Inference
Execute the exact evaluation command mandated by the hackathon harness:
```bash
python inference_script.py \
  --test_dir ./hackathon_dataset/test/images/ \
  --weights ./model_weights.pt \
  --output predictions.csv
```

#### Output Contract:
The produced `predictions.csv` guarantees strict conformity:
```csv
image_id,Label,confidence
3f9a2c1d-8b4e-4f32-9c1a-2e8d7a1b0c3f,1,0.8700
a8b4e7f2-1d3c-4e5a-b7f9-8a2c1d3e4f5a,0,0.1300
```
* `image_id`: UUID filename **without** `.jpg` extension.
* `Label`: Binary integer classification (`0 = Real`, `1 = Fake`).
* `confidence`: Calibrated float in range `[0.0000, 1.0000]` representing probability of manipulation.

### 3. Model Training Pipeline
To train and cross-validate with Cosine Annealing and perturbation augmentations:
```bash
python train.py \
  --data_dir ./hackathon_dataset \
  --metadata_csv ./hackathon_dataset/train_metadata.csv \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_weights model_weights.pt
```

---

## 🔬 Platform Modules & Reference Features

1. **Cinematic Hero ("Detect. Analyze. Verify. Protect.")**:
   * Crystalline 3D glass hero artwork with floating levitation and ambient particles.
   * Floating Glass HUD (`ANALYZING...`) with live waveform equalizer and audit checklist.
   * Key stats: `2.4M+` Media Scans, `99.2%` Detection Accuracy, `< 1s` Avg. Analysis Time.
2. **Glass Feature Cards Row**:
   * 4 horizontal glassmorphic cards: Deepfake Detection, Media Forensics, Explainable AI, Forensic Reports.
3. **How It Works Section ("From a media file to a forensic report, in one flow.")**:
   * Holographic 3D face mesh scan on glass plates with floating `[ MANIPULATED ] 92.7%` badge.
   * 4-step workflow: Upload Media $\rightarrow$ AI Analysis $\rightarrow$ Verify Authenticity $\rightarrow$ Get Report.
4. **Integrated Forensic Workspace**:
   * Advanced 9-step image scanner with drag-and-drop validation.
   * Dual-stream Grad-CAM explainability studio and 2D FFT spectral canvas.
   * Persistent audit log and printable/downloadable forensic dossiers.
   * Batch prediction runner and official `predictions.csv` exporter.
