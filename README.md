# SkinCancerViT: A Multimodal Deep Learning Approach for Skin Cancer Classification

## Project Overview

SkinCancerViT is a deep learning project focused on the early classification of skin cancer.

Leveraging the power of multimodal data, this model aims to assist dermatologists by providing diagnostic predictions, integrating visual and clinical information.

## Problem

Accurate and timely diagnosis of skin cancer is crucial for effective treatment and improved patient outcomes.

Traditional methods can be subjective, and relying solely on visual inspection or image analysis may miss critical clinical cues.

## Solution

SkinCancerViT addresses this challenge by employing a multimodal deep learning architecture that combines:

- **Dermoscopic Image Analysis**: Utilizes a Vision Transformer (ViT) backbone, pre-trained on a large image dataset, to extract rich visual features from skin lesion images.
- **Tabular Clinical Data Integration**: Incorporates essential patient metadata, specifically age and lesion localization, processed through a dedicated Multi-Layer Perceptron (MLP).

These two distinct data modalities are fused to create a comprehensive representation, allowing the model to make more informed and robust predictions.

## Dataset

The model was trained and evaluated on the `marmal88/skin_cancer` dataset, a collection of skin lesion images paired with associated clinical metadata.

The training utilized ~10k records.

## Installation

First, clone the repository:

```bash
git clone https://github.com/ethicalabs-ai/SkinCancerViT.git
cd SkinCancerViT
```

Then, install the package in editable mode using uv (or pip):

```bash
uv sync   # Recommended if you use uv
# Or, if using pip:
# pip install -e .
```

## Quick Start / Usage

This package allows you to load and use a pre-trained SkinCancerViT model for prediction.

```python
import torch
from skincancer_vit.model import SkinCancerViTModel
from PIL import Image
from datasets import load_dataset   # To get a random sample

# Load the model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerViTModel.from_pretrained("ethicalabs/SkinCancerViT")
model.to(device)   # Move model to the desired device
model.eval()   # Set model to evaluation mode

# Example Prediction from a Specific Image File
image_file_path = "images/patient-001.jpg"   # Specify your image file path here
specific_image = Image.open(image_file_path).convert("RGB")

# Example tabular data for this prediction
specific_age = 42
specific_localization = "face"   # Ensure this matches one of your trained localization categories

predicted_dx, confidence = model.full_predict(
    raw_image=specific_image,
    raw_age=specific_age,
    raw_localization=specific_localization,
    device=device
)

print(f"Predicted Diagnosis: {predicted_dx}")
print(f"Confidence: {confidence:.4f}")

# Example Prediction from a Random Test Sample from the Dataset
dataset = load_dataset("marmal88/skin_cancer", split="test")
random_sample = dataset.shuffle(seed=42).select(range(1))[0] # Get the first shuffled sample

sample_image = random_sample["image"]
sample_age = random_sample["age"]
sample_localization = random_sample["localization"]
sample_true_dx = random_sample["dx"]

predicted_dx_sample, confidence_sample = model.full_predict(
    raw_image=sample_image,
    raw_age=sample_age,
    raw_localization=sample_localization,
    device=device
)

print(f"Predicted Diagnosis: {predicted_dx_sample}")
print(f"Confidence: {confidence_sample:.4f}")
print(f"Correct Prediction: {predicted_dx_sample == sample_true_dx}")
```

## Key Achievements & Performance

The SkinCancerViT model achieved **97.35% success rate** on 1285 test samples, demonstrating its exceptional capability in distinguishing between various skin cancer diagnoses.

This high performance underscores the effectiveness of the multimodal approach in capturing complex patterns from both visual and clinical data.

## Technical Highlights

- **Custom Multimodal Architecture**: A `SkinCancerViTModel` class built on `PreTrainedModel`, seamlessly integrating a `transformers` Vision Transformer with custom tabular processing layers.
- **Hugging Face PreTrainedModel Compatibility**: The `SkinCancerViTModel` is designed to be compatible with the Hugging Face `PreTrainedModel` API, enabling easy saving (`save_pretrained`) and loading (`from_pretrained`) of the entire model, including its configuration and weights.
- **Efficient Data Handling**: Utilizes `transformers.AutoImageProcessor` for consistent image preprocessing and a custom data collator for efficient batching of multimodal inputs.

## Disclaimer

This project is a work in progress and is purely experimental.

The SkinCancerViT model is developed with the primary goal of evaluating the capabilities of Vision Transformer models in multimodal medical image analysis.

**It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.**

The predictions generated by this model are for research and informational purposes only.

Every diagnosis should be confirmed by a qualified medical professional, and a biopsy remains the gold standard for definitive skin cancer diagnosis.