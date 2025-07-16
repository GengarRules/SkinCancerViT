import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image
import os
from safetensors.torch import load_file  # Import load_file for .safetensors

# --- Import necessary components from main.py ---
try:
    from main import CombinedModel, load_and_prepare_data, create_preprocessing_function
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    print(
        "Please ensure main.py is in the same directory and defines CombinedModel, load_and_prepare_data, and create_preprocessing_function."
    )
    exit()


# --- New: Easy-to-use SkinCancerPredictor Class ---
class SkinCancerPredictor:
    def __init__(
        self,
        model_path,
        model_checkpoint_name,
        id2label,
        localization_to_id,
        num_localization_features,
        normalize_age,
        total_tabular_features_dim,
    ):
        """
        Initializes the SkinCancerPredictor.

        Args:
            model_path (str): Path to the directory containing the saved model (e.g., ./skin_cancer_multimodal_results/final_model).
                              This directory should contain 'model.safetensors'.
            model_checkpoint_name (str): The name of the Hugging Face model checkpoint used for the vision backbone
                                         (e.g., "google/vit-base-patch16-224-in21k").
            id2label (dict): A dictionary mapping label IDs to human-readable diagnosis names.
            localization_to_id (dict): A dictionary mapping localization names to their one-hot encoding IDs.
            num_localization_features (int): The total number of unique localization features.
            normalize_age (function): The function used to normalize age values.
            total_tabular_features_dim (int): The total dimension of the tabular features (localization + age).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_name)

        # Initialize model architecture
        self.model = CombinedModel(
            model_checkpoint_name, total_tabular_features_dim, len(id2label)
        )

        # --- Robust Model Loading for .safetensors ---
        model_sf_path = os.path.join(
            model_path, "model.safetensors"
        )  # Changed to .safetensors
        if not os.path.exists(model_sf_path):
            raise FileNotFoundError(
                f"Model file not found: {model_sf_path}. Please ensure it exists."
            )

        # Check if the file is empty or too small (a common sign of corruption)
        if (
            os.path.getsize(model_sf_path) < 1024
        ):  # Less than 1KB, likely corrupted or empty
            raise ValueError(
                f"Model file '{model_sf_path}' is too small ({os.path.getsize(model_sf_path)} bytes). It might be corrupted or incomplete."
            )

        try:
            # Use safetensors.torch.load_file to load the state dictionary
            model_state_dict = load_file(
                model_sf_path, device=str(self.device)
            )  # device must be string for safetensors
            self.model.load_state_dict(model_state_dict)
            self.model.eval()  # Set model to evaluation mode
            self.model.to(self.device)  # Move model to appropriate device
            print(
                "Trained model weights loaded successfully and model set to evaluation mode."
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading model state dictionary from '{model_sf_path}'. "
                f"This often indicates a corrupted or invalid .safetensors file. "
                f"Original error: {e}"
            )

        # Store mappings and functions
        self.id2label = id2label
        self.localization_to_id = localization_to_id
        self.num_localization_features = num_localization_features
        self.normalize_age = normalize_age

    def predict(self, pil_image: Image.Image, age: int, localization: str) -> str:
        """
        Performs inference on a single image and tabular data.

        Args:
            pil_image (PIL.Image.Image): The input skin lesion image (PIL Image object).
            age (int): The patient's age.
            localization (str): The lesion's localization (e.g., "face", "back").

        Returns:
            str: The predicted diagnosis label.
        """
        # 1. Preprocess Image
        img_rgb = pil_image.convert("RGB")
        processed_img = self.image_processor(img_rgb, return_tensors="pt").to(
            self.device
        )
        pixel_values = processed_img[
            "pixel_values"
        ]  # Already has batch dimension (1, C, H, W)

        # 2. Preprocess Tabular Features
        localization_one_hot = torch.zeros(
            self.num_localization_features, device=self.device
        )
        if localization in self.localization_to_id:
            localization_one_hot[self.localization_to_id[localization]] = 1.0

        age_normalized = torch.tensor(
            [self.normalize_age(age)], dtype=torch.float, device=self.device
        )

        tabular_features = torch.cat([localization_one_hot, age_normalized]).unsqueeze(
            0
        )  # Add batch dimension

        # 3. Perform Inference
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values, tabular_features=tabular_features
            )
            logits = outputs["logits"]
            predicted_class_id = torch.argmax(logits, dim=-1).item()

        return self.id2label[predicted_class_id]


# --- Main Inference Execution Block ---
if __name__ == "__main__":
    # Define the path to the explicitly saved final model
    model_path = "./skin_cancer_multimodal_results/final_model"

    # Check if the final model directory exists
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        print(f"Error: Final model directory '{model_path}' not found.")
        print(
            "Please ensure you have run main.py with the updated code to save the final model."
        )
        exit()

    print(f"Loading model from: {model_path}")

    # Step 1: Load and Prepare Data (to get mappings and a sample for inference)
    # Use load_and_prepare_data from main.py to get the mappings (label2id, id2label, etc.)
    # We load the full dataset to ensure all mappings are consistent with training.
    # The 'dataset' returned here is the full dataset split into train/val/test,
    # but we only need the mappings from it for the predictor.
    (
        full_dataset_for_mappings,
        label2id,
        id2label,
        num_dx_labels,
        localization_to_id,
        num_localization_features,
        normalize_age,
        total_tabular_features_dim,
    ) = load_and_prepare_data(num_records_to_use=10000)

    # Step 2: Initialize the SkinCancerPredictor
    model_checkpoint_name = "google/vit-base-patch16-224-in21k"
    try:
        predictor = SkinCancerPredictor(
            model_path=model_path,
            model_checkpoint_name=model_checkpoint_name,
            id2label=id2label,
            localization_to_id=localization_to_id,
            num_localization_features=num_localization_features,
            normalize_age=normalize_age,
            total_tabular_features_dim=total_tabular_features_dim,
        )
        print("\nSkinCancerPredictor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize SkinCancerPredictor: {e}")
        exit()

    # Step 3: Demonstrate Inference with a few samples from the test set
    print("\nDemonstrating inference on a few test samples:")

    # Get a few raw examples from the test split for demonstration
    # We explicitly load the test split here, separate from the full_dataset_for_mappings
    test_samples_raw = (
        load_dataset("marmal88/skin_cancer", split="test")
        .shuffle(seed=42)
        .select(range(1282))
    )

    correct_predictions = 0
    total_samples_demonstrated = len(test_samples_raw)

    for i, example in enumerate(test_samples_raw):
        pil_image = example["image"]
        age = example["age"]
        localization = example["localization"]
        true_dx = example["dx"]

        print(f"\n--- Sample {i+1} ---")
        print(
            f"Input: Age={age}, Localization='{localization}', True Diagnosis='{true_dx}'"
        )

        try:
            predicted_dx = predictor.predict(pil_image, age, localization)
            print(f"Predicted Diagnosis: '{predicted_dx}'")
            if predicted_dx == true_dx:
                print("Prediction: CORRECT")
                correct_predictions += 1
            else:
                print("Prediction: INCORRECT")
        except Exception as e:
            print(f"Error during prediction for sample {i+1}: {e}")

    print("\nInference demonstration complete.")

    # Calculate and print success percentage
    if total_samples_demonstrated > 0:
        success_percentage = (correct_predictions / total_samples_demonstrated) * 100
        print(
            f"Demonstration Success Rate: {success_percentage:.2f}% ({correct_predictions}/{total_samples_demonstrated} correct)"
        )
    else:
        print("No samples were processed for demonstration.")
