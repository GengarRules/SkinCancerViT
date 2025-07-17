import torch
from datasets import load_dataset
import os

from skincancer_vit.model import SkinCancerViTModel


if __name__ == "__main__":
    # Define the path to the explicitly saved final model
    model_path = "./results/final_model"

    # Check if the final model directory exists and contains necessary files
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        print(f"Error: Final model directory '{model_path}' not found.")
        print(
            "Please ensure you have run main.py with the updated code to save the final model."
        )
        exit()
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(
            f"Error: 'config.json' not found in '{model_path}'. Please run main.py to save the model correctly."
        )
        exit()
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(
            f"Error: 'model.safetensors' not found in '{model_path}'. Please run train.py to save the model correctly."
        )
        exit()

    print(f"Loading model from: {model_path}")

    # Determine the device to use for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained SkinCancerViTModel using from_pretrained
    try:
        model = SkinCancerViTModel.from_pretrained(model_path)
        print(
            f"Trained model loaded successfully from '{model_path}' using from_pretrained."
        )
    except Exception as e:
        print(
            f"Error loading model from '{model_path}' using from_pretrained: {e}. "
            f"Ensure 'config.json' and 'model.safetensors' are present and valid."
        )
        exit()

    print("\nDemonstrating inference on a few test samples:")

    # Get a few raw examples from the test split for test
    test_samples_raw = (
        load_dataset("marmal88/skin_cancer", split="test")
        .shuffle(seed=42)
        .select(range(1000))
    )

    correct_predictions = 0
    total_samples_demonstrated = len(test_samples_raw)

    for i, example in enumerate(test_samples_raw):
        pil_image = example["image"]
        age = example["age"]
        localization = example["localization"]
        true_dx = example["dx"]

        print(f"\n--- Sample {i + 1} ---")
        print(
            f"Input: Age={age}, Localization='{localization}', True Diagnosis='{true_dx}'"
        )

        try:
            # Use the model's full_predict method directly
            # This method handles all preprocessing, inference, and postprocessing internally.
            predicted_dx, predicted_confidence = model.full_predict(
                raw_image=pil_image,
                raw_age=age,
                raw_localization=localization,
                device=device,  # Pass the device to the full_predict method
            )
            # full_predict returns a single label string and a single float confidence
            # So, no need to extract from lists or convert ID to string.
            predicted_dx_label = predicted_dx
            predicted_confidence_score = predicted_confidence

            print(
                f"Predicted Diagnosis: '{predicted_dx_label}' (Confidence: {predicted_confidence_score:.4f})"
            )
            if predicted_dx_label == true_dx:
                print("Prediction: CORRECT")
                correct_predictions += 1
            else:
                print("Prediction: INCORRECT")
        except Exception as e:
            print(f"Error during prediction for sample {i + 1}: {e}")

    print("\nInference demonstration complete.")

    # Calculate and print success percentage
    if total_samples_demonstrated > 0:
        success_percentage = (correct_predictions / total_samples_demonstrated) * 100
        print(
            f"Demonstration Success Rate: {success_percentage:.2f}% ({correct_predictions}/{total_samples_demonstrated} correct)"
        )
    else:
        print("No samples were processed for demonstration.")
