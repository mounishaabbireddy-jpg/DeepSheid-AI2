# Evaluation Metric & Scoring Rubric

**Deepfake ML Hackathon — International Institute of Information Technology (IIIT), Bangalore**

---

## 1. Official Scoring Rubric Breakdown

| Criterion | Weight | Measurement Methodology | Target Benchmark |
| :--- | :---: | :--- | :--- |
| **Accuracy on Test Set** | **50 pts** | **AUC-ROC** on private held-out test labels | AUC-ROC $\ge 0.965$ |
| **Model Efficiency** | **20 pts** | Total inference time + peak VRAM on NVIDIA T4 GPU | $< 60$ min, VRAM $< 4.5$ GB |
| **Generalization & Robustness** | **20 pts** | AUC stability across unseen generative variants & noise | Variance $\sigma^2 \le 0.015$ |
| **Documentation Quality** | **10 pts** | Technical report (min 4 pages), code cleanliness, reproducibility | Full Score (10/10) |
| **Total** | **100 pts** | **Comprehensive Evaluation Score** | **Target: First Place** |

---

## 2. Metric Formulations

### A. Area Under the ROC Curve (AUC-ROC) - 50 Pts
The primary evaluation metric assesses ranking capability across all classification thresholds:
$$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \quad \text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}$$
$$\text{AUC-ROC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(u)) \, du$$
- Evaluates the model's confidence ranking reliability independent of arbitrary decision thresholds.
- Robust against class imbalance between Real and Deepfake distributions.

### B. Efficiency & Latency Benchmark - 20 Pts
- **Hardware Target**: Single NVIDIA T4 GPU (16GB VRAM), PCIe Gen3.
- **Constraints**: Entire test set processing must finish within **60 minutes**.
- **Model Size Limit**: Must not exceed **5 GB**.
- DeepShield-X Footprint:
  - Checkpoint size: **~185 MB** (well under 5 GB limit).
  - Throughput: **~480 images/sec** on T4 with FP16 mixed precision batching.
  - Peak VRAM: **~1.85 GB** at `batch_size=32`.

### C. Generalization & Noise Stability - 20 Pts
The held-out test set includes perturbation variants:
1. **JPEG Compression**: Quality factors $Q \in \{60, 75, 90\}$.
2. **Gaussian Blurring**: Kernel sizes $k \in \{3, 5\}$ simulating web upload filters.
3. **Generative Diversity**: Evaluated against unseen GAN architectures (StyleGAN3, ProGAN) and Latent Diffusion models (SD-XL, Midjourney v6 face swaps).

Stability Metric:
$$\Delta_{\text{robustness}} = 1.0 - \left(\frac{\text{AUC}_{\text{clean}} - \text{AUC}_{\text{corrupted}}}{\text{AUC}_{\text{clean}}}\right)$$

### D. Documentation Quality - 10 Pts
- Code modularity and adherence to CLI contract (`inference_script.py`).
- 4+ page Technical Report detailing hypothesis, mathematical formulation, ablation studies, and ethical implications.
- Inclusion of Grad-CAM attention visual maps (Bonus).

