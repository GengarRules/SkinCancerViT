import torch
from datasets import load_dataset
from transformers import AutoImageProcessor
import os

from skincancer_vit.model import SkinCancerViTModel
from skincancer_vit.data import load_and_prepare_data, create_preprocessing_function


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

    # Step 1: Load and Prepare Data (to get mappings for preprocessing)
    # Use load_and_prepare_data from main.py to get the mappings (label2id, id2label, etc.)
    # We load the full dataset to ensure all mappings are consistent with training.
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

    # Determine the device to use for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load Image Processor
    model_checkpoint_name = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint_name)
    print("Image processor loaded.")

    # Step 3: Load the trained SkinCancerViTModel using from_pretrained
    try:
        model = SkinCancerViTModel.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode
        model.to(device)  # Move model to appropriate device
        print(
            f"Trained model loaded successfully from '{model_path}' using from_pretrained and set to evaluation mode."
        )
    except Exception as e:
        print(
            f"Error loading model from '{model_path}' using from_pretrained: {e}. "
            f"Ensure 'config.json' and 'model.safetensors' are present and valid."
        )
        exit()

    # Create Preprocessing Function
    preprocess_function_inference = create_preprocessing_function(
        image_processor,
        label2id,
        localization_to_id,
        num_localization_features,
        normalize_age,
    )

    print("\nDemonstrating inference on a few test samples:")

    # Get a few raw examples from the test split for test
    test_samples_raw = (
        load_dataset("marmal88/skin_cancer", split="test")
        .shuffle(seed=42)
        .select(range(1000))
    )

    correct_predictions = 0
    total_samples_demonstrated = len(test_samples_raw)

    with torch.no_grad():  # Disable gradient calculation for inference
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
                # Preprocess Image
                img_rgb = pil_image.convert("RGB")
                processed_img = image_processor(img_rgb, return_tensors="pt").to(device)
                pixel_values = processed_img[
                    "pixel_values"
                ]  # Already has batch dimension (1, C, H, W)

                # Preprocess Tabular Features
                localization_one_hot = torch.zeros(
                    num_localization_features, device=device
                )
                if localization in localization_to_id:
                    localization_one_hot[localization_to_id[localization]] = 1.0

                age_normalized = torch.tensor([age], dtype=torch.float, device=device)

                tabular_features = torch.cat(
                    [localization_one_hot, age_normalized]
                ).unsqueeze(0)  # Add batch dimension

                # Perform Inference
                outputs = model(
                    pixel_values=pixel_values, tabular_features=tabular_features
                )
                logits = outputs["logits"]
                predicted_class_id = torch.argmax(logits, dim=-1).item()
                predicted_dx = id2label[predicted_class_id]

                print(f"Predicted Diagnosis: '{predicted_dx}'")
                if predicted_dx == true_dx:
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
