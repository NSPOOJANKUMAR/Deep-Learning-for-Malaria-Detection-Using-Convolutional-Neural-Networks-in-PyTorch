# Deep Learning for Malaria Detection Using Convolutional Neural Networks in PyTorch

**Student:** M N S Poojan Kumar, U00954242

**Dataset:** NIH Malaria Cell Images Dataset (National Institutes of Health, via Kaggle)


## Overview

This project implements an automated malaria detection pipeline using convolutional neural networks in PyTorch. Five architectures are trained and evaluated on the NIH Malaria Cell Images Dataset: a custom CNN built from random initialisation, and four transfer learning models (ResNet-18, VGG16, ResNet-50, EfficientNet-B0). The primary clinical objective is to maximise recall on the Parasitized class, as false negatives represent missed diagnoses with severe clinical consequences.


## Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Parameters |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 97.79% | 97.79% | 97.79% | 97.79% | 1.00 | ~5.3M |
| ResNet-50 | 97.79% | 97.79% | 97.79% | 97.79% | 1.00 | ~25.6M |
| VGG16 | 97.50% | 97.50% | 97.50% | 97.50% | 1.00 | ~138M |
| ResNet-18 | 97.32% | 97.33% | 97.32% | 97.32% | 1.00 | ~11.7M |
| Custom CNN | 96.55% | 96.56% | 96.55% | 96.55% | 0.99 | ~2.5M |

**Recommended model:** EfficientNet-B0 achieves the highest accuracy and recall (97.79%) with the smallest parameter count among the transfer learning models (~5.3M), making it the most efficient option for deployment.

---

## How to Run

### Requirements
- Python 3.8 or later
- GPU strongly recommended (Google Colab T4 or equivalent)

### Step 1: Open the Notebook

Open `malaria_final_notebook.ipynb` in Google Colab.

Enable GPU: Runtime > Change runtime type > T4 GPU.

### Step 2: Install Dependencies

Cell 1a installs all packages automatically. To install manually:

```bash
pip install torch torchvision kagglehub scikit-learn seaborn opencv-python matplotlib Pillow tqdm pandas
```

### Step 3: Configure Kaggle Credentials (Local Only)

When running outside of Colab, place a valid `kaggle.json` API key in `~/.kaggle/`. KaggleHub manages credentials automatically within Colab.

### Step 4: Run All Cells in Order

Execute cells sequentially from top to bottom. Do not skip cells, as each section depends on outputs from preceding ones.

Use **Runtime > Run all** (Ctrl+F9) in Colab to execute the complete notebook automatically.

---

## Expected Output

| Output | Description |
|---|---|
| `fig_cm_*.png` | Confusion matrix visualisations for each model |
| `fig_training_*.png` | Training and validation loss and accuracy curves |
| `fig_all5_comparison.png` | Side-by-side performance comparison chart |
| `fig_gradcam.png` | Grad-CAM activation maps for the best model |

Training runs produce per-epoch loss and accuracy logs for all five architectures. Total runtime on a Colab T4 GPU is approximately 35 to 45 minutes for all five models.

---

## File Structure

```
malaria_final_notebook.ipynb   Main Jupyter notebook (complete pipeline)
malaria_final_report.docx      Final project report with charts and discussion
README.md                      This file
```

---

## Key Design Decisions

**Unified training function:** A single `train_model()` function handles all five architectures, eliminating duplicated training code and ensuring identical experimental conditions across models.

**Early stopping:** Training terminates when validation loss fails to improve by at least 0.001 for four consecutive epochs, preventing wasted computation and reducing overfitting risk.

**Separate learning rates:** The Custom CNN uses a learning rate of 1e-3 from random initialisation. All pretrained models use 1e-4 to preserve ImageNet feature representations during fine-tuning.

**80/10/10 data split:** The test set is accessed exactly once, after all model training is complete, to provide an unbiased performance estimate.

**Seed 42:** All random operations are seeded for full reproducibility across independent runs on identical hardware.

---

## Ethical Considerations

This system is a clinical decision-support tool designed to assist qualified healthcare professionals. It is not intended to function as an autonomous diagnostic instrument. All model predictions in a real clinical deployment must be reviewed by a trained clinician before any diagnostic or treatment decision is made.

The model was developed and evaluated exclusively on the NIH Malaria Cell Images benchmark dataset. Performance on images collected from different staining protocols, microscope platforms, or patient populations has not been established.

---

## Citations

- NIH Malaria Cell Images Dataset. Kaggle (iarunava). National Institutes of Health.
- Paszke et al. (2019). PyTorch: An Imperative Style High-Performance Deep Learning Library. NeurIPS 2019.
- Tan and Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML 2019.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV 2017.
- AI assistance: Claude (Anthropic) was used to assist with code structure, markdown content, and report writing. All model architectures, training procedures, and experimental results represent the original work of the student.
