import gradio as gr
import torch
from PIL import Image
from datasets import load_dataset
import random

from skincancer_vit.model import SkinCancerViTModel

HF_MODEL_REPO = "ethicalabs/SkinCancerViT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Loading SkinCancerViT model from {HF_MODEL_REPO} to {DEVICE}...")

model = SkinCancerViTModel.from_pretrained(HF_MODEL_REPO)
model.to(DEVICE)
model.eval()  # Set to evaluation mode
print("Model loaded successfully.")

print("Loading 'marmal88/skin_cancer' dataset for random samples...")
dataset = load_dataset("marmal88/skin_cancer", split="test")
print("Dataset loaded successfully.")


def predict_uploaded_image(image: Image.Image, age: int, localization: str) -> str:
    """
    Handles prediction for an uploaded image with user-provided tabular data.
    """
    if model is None:
        return "Error: Model not loaded. Please check the console for details."
    if image is None:
        return "Please upload an image."
    if age is None:
        return "Please enter an age."
    if not localization:
        return "Please select a localization."

    try:
        # Call the model's full_predict method
        predicted_dx, confidence = model.full_predict(
            raw_image=image, raw_age=age, raw_localization=localization, device=DEVICE
        )
        return f"Predicted Diagnosis: **{predicted_dx}** (Confidence: {confidence:.4f})"
    except Exception as e:
        return f"Prediction Error: {e}"


# --- Prediction Function for Random Sample ---
def predict_random_sample() -> str:
    """
    Fetches a random sample from the dataset and performs prediction.
    """
    if model is None:
        return "Error: Model not loaded. Please check the console for details."
    if dataset is None:
        return "Error: Dataset not loaded. Cannot select random sample."

    try:
        # Select a random sample from the dataset
        random_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[random_idx]

        sample_image = sample["image"]
        sample_age = sample["age"]
        sample_localization = sample["localization"]
        sample_true_dx = sample["dx"]

        # Call the model's full_predict method
        predicted_dx, confidence = model.full_predict(
            raw_image=sample_image,
            raw_age=sample_age,
            raw_localization=sample_localization,
            device=DEVICE,
        )

        # Return a formatted string with all information
        result_str = (
            f"**Random Sample Details:**\n"
            f"- Age: {sample_age}\n"
            f"- Localization: {sample_localization}\n"
            f"- True Diagnosis: **{sample_true_dx}**\n\n"
            f"**Model Prediction:**\n"
            f"- Predicted Diagnosis: **{predicted_dx}**\n"
            f"- Confidence: {confidence:.4f}\n"
            f"- Correct Prediction: {'✅ Yes' if predicted_dx == sample_true_dx else '❌ No'}"
        )
        return sample_image, result_str
    except Exception as e:
        return None, f"Prediction Error on Random Sample: {e}"


# --- Gradio Interface ---
with gr.Blocks(title="Skin Cancer ViT Predictor") as demo:
    gr.Markdown(
        """
        # Skin Cancer ViT Predictor
        This application demonstrates the `SkinCancerViT` multimodal model for skin cancer diagnosis.
        It can take an uploaded image with patient metadata or predict on a random sample from the dataset.
        **Disclaimer:** This tool is for demonstration and research purposes only and should not be used for medical diagnosis.
        """
    )

    with gr.Tab("Predict on Random Sample"):
        gr.Markdown("## Get a Prediction from a Random Sample in the Test Set")
        random_sample_button = gr.Button("Get Random Sample Prediction")

        # Modified output components for random sample tab
        with gr.Row():
            output_random_image = gr.Image(
                type="pil", label="Random Sample Image", height=250, width=250
            )
            output_random_details = gr.Markdown(
                "Random sample details and prediction will appear here."
            )

        random_sample_button.click(
            fn=predict_random_sample,
            inputs=[],
            outputs=[
                output_random_image,
                output_random_details,
            ],  # Map to both image and markdown outputs
        )

    with gr.Tab("Upload Image & Predict"):
        gr.Markdown("## Upload Your Image and Provide Patient Data")
        with gr.Row():
            image_input = gr.Image(
                type="pil", label="Upload Skin Lesion Image (224x224 preferred)"
            )
            with gr.Column():
                age_input = gr.Number(
                    label="Patient Age", minimum=0, maximum=120, step=1
                )
                # Ensure these localizations match your training data categories
                localization_input = gr.Dropdown(
                    model.config.localization_to_id.keys(),
                    label="Lesion Localization",
                    value="unknown",  # Default value
                )
                predict_button = gr.Button("Get Prediction")

        output_upload = gr.Markdown("Prediction will appear here.")

        predict_button.click(
            fn=predict_uploaded_image,
            inputs=[image_input, age_input, localization_input],
            outputs=output_upload,
        )

if __name__ == "__main__":
    demo.launch(share=False)
