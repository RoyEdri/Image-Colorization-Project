# Image Colorization with Conditional GANs (Pix2Pix)

### 🎯 Project Overview

This project addresses **automatic image colorization** of grayscale bird images using the *20 UK Garden Birds* dataset from Kaggle.
We implemented and compared two approaches:

1. **Naive CNN Autoencoder** – a supervised baseline for grayscale→RGB translation.
2. **Conditional GAN (Pix2Pix)** – using a **U-Net generator** for stable, fine-grained generation and a **PatchGAN discriminator** for local realism.

The goal was to evaluate how **adversarial training** improves realism and sharpness compared to traditional supervised models.

---

## 📚 Research Context

The project builds on the paper

> *Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)*

Pix2Pix introduced the idea of **conditional GANs (cGANs)** for pixel-to-pixel mapping tasks such as grayscale→color, edges→photo, etc.
Its loss combines a **GAN component** (realism) with an **L1 reconstruction term** (structural accuracy):
[
L = L_{cGAN}(G,D) + \lambda L_{1}(G)
]
where **λ = 100** balances color consistency and sharpness.

---

## 🧠 Implementation Details

### Dataset

* **Source:** [20 UK Garden Birds](https://www.kaggle.com/datasets/davemahony/20-uk-garden-birds)
* **Input:** Grayscale image (128×128×1)
* **Output:** RGB image (128×128×3)
* **Split:** 80% train, 10% validation, 10% test
* **Preprocessing:** resizing, normalization (0–1 or −1–1), controlled augmentation (flips, rotations)

---

### Architectures

#### 🧩 1. Naive CNN Autoencoder

* Encoder–decoder CNN
* Experiments:

  * **Exp.1:** Baseline (simple CNN)
  * **Exp.2:** + Dropout (reduce overfitting)
  * **Exp.3:** + BatchNorm & deeper network (improved stability and color accuracy)
* **Loss:** MSE   |   **Optimizer:** Adam (1e-3)

#### ⚡ 2. Conditional GAN (Pix2Pix)

* **Generator:** U-Net (encoder-decoder with skip connections)
* **Discriminator:** PatchGAN (70×70 local classifier)
* **Loss:** L1 + GAN (λ = 100)
* **Optimizer:** Adam (2e-4, β1 = 0.5)
* **Experiments:**

  * **Exp.1:** Baseline Pix2Pix (L1 + GAN)
  * **Exp.2:** MobileNetV2 encoder integration (partial unfreezing ≈20 layers)
  * **Exp.3:** Tuned λ values (80–120) to balance realism vs. consistency

---

## 🧪 Experiments and Results

| Model                     | Key Techniques       | Best Epoch | Observations                          |
| :------------------------ | :------------------- | :--------: | :------------------------------------ |
| CNN Baseline              | Simple Autoencoder   |     20     | Low-saturation colors, underfitting   |
| CNN + Dropout             | Regularization (0.2) |     50     | Less overfitting, slightly blurry     |
| CNN + BatchNorm (deeper)  | Stable training      |     70     | Sharper colors, best CNN result       |
| Pix2Pix (L1 + GAN, λ=100) | U-Net + PatchGAN     |     50     | Strong color realism, minor artifacts |
| Pix2Pix + MobileNetV2     | Transfer learning    |    100+    | Richer colors, risk of overfitting    |

**Performance Summary**

* BatchNorm CNN → best supervised result.
* Pix2Pix (λ=100) → most vivid and realistic outputs.
* MobileNetV2 encoder enhanced texture detail but required careful fine-tuning.
* Qualitative evaluation (output images) was more informative than numeric loss values.

---

## 🧩 Notebook Structure

```
├── Notebook 1 – Naive_CNN_Colorization.ipynb
│   ├── Data loading & preprocessing
│   ├── 3 CNN experiments (baseline → BatchNorm)
│   └── Loss curves + generated outputs
│
├── Notebook 2 – Pix2Pix_CGAN_Colorization.ipynb
│   ├── U-Net & PatchGAN implementation
│   ├── 3 CGAN experiments (λ, MobileNetV2)
│   └── Visual comparisons & training plots
│
└── Notebook 3 – Test_Environment.ipynb
    ├── Load best weights
    ├── Inference on new images
    └── Display side-by-side results
```

---

## ⚙️ Setup & Execution

1. **Environment**

   * Python ≥ 3.9
   * TensorFlow/Keras ≥ 2.10
   * NumPy, Matplotlib, gdown, PIL

2. **Run in Google Colab**

   * Mount Google Drive
   * Download dataset and weights (via `gdown`)
   * Execute cells sequentially

3. **Inference**

   * Upload a 128×128 grayscale image
   * Model outputs its colorized version

---

## 💡 Insights & Future Work

* **L1 vs. GAN balance:** λ≈100 offered the best trade-off between global color accuracy and sharpness.
* **Skip connections:** Improved luminance preservation and detail fidelity.
* **Transfer learning:** MobileNetV2 encoder added semantic context but required layer-freezing control.
* **Next steps:** Try perceptual loss (VGG), larger datasets, and Vision Transformer backbones for richer color semantics.

---

## 👥 Team

**Roy Edri**, **Nati Forish**, **Iyar Gadulov**, **Inon Elgabsi**
*Advanced Topics in Deep Learning – Final Project (2025)*
