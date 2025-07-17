import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    AutoImageProcessor,
)
from typing import Any, Tuple


class SkinCancerViTModelConfig(PretrainedConfig):
    model_type = "combined_multimodal_skin_cancer"

    def __init__(
        self,
        vision_model_checkpoint="google/vit-base-patch16-224-in21k",
        total_tabular_features_dim=None,
        num_dx_labels=None,
        id2label=None,  # Dictionary mapping ID to label string
        label2id=None,  # Dictionary mapping label string to ID
        localization_to_id=None,  # Dictionary mapping localization string to ID
        num_localization_features=None,  # Total number of localization categories
        age_mean=None,  # Mean for age normalization
        age_std=None,  # Standard deviation for age normalization
        age_min=None,  # Min age for age normalization
        age_max=None,  # Min age for age normalization
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model_checkpoint = vision_model_checkpoint
        self.total_tabular_features_dim = total_tabular_features_dim
        self.num_dx_labels = num_dx_labels
        self.id2label = id2label
        self.label2id = label2id
        self.localization_to_id = localization_to_id
        self.num_localization_features = num_localization_features
        self.age_mean = age_mean
        self.age_std = age_std
        self.age_min = age_min
        self.age_max = age_max


class SkinCancerViTModel(PreTrainedModel):
    config_class = SkinCancerViTModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config  # Store config for easy access

        # Load the Vision Transformer backbone using config.vision_model_checkpoint
        self.vision_model = AutoModel.from_pretrained(config.vision_model_checkpoint)
        self.vision_output_dim = self.vision_model.config.hidden_size

        # MLP for tabular features using config.total_tabular_features_dim
        self.tabular_mlp = nn.Sequential(
            nn.Linear(
                config.total_tabular_features_dim, 128
            ),  # Smaller hidden layer for tabular
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # Output dimension for tabular features
        )
        self.tabular_output_dim = 64

        # Final classification head using config.num_dx_labels
        self.classifier = nn.Linear(
            self.vision_output_dim + self.tabular_output_dim, config.num_dx_labels
        )

        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()

        # This calls PreTrainedModel's post_init which handles default initialization
        self.post_init()

        self.image_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_checkpoint
        )

    def forward(self, pixel_values, tabular_features, labels=None):
        # Process image data
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # Get the pooled output (usually the CLS token's representation)
        vision_pooled_output = (
            vision_outputs.pooler_output
            if vision_outputs.pooler_output is not None
            else vision_outputs.last_hidden_state[:, 0]
        )

        # Process tabular data
        # Ensure tabular_features are float type for the MLP
        tabular_output = self.tabular_mlp(tabular_features.float())

        # Concatenate vision and tabular features
        combined_features = torch.cat((vision_pooled_output, tabular_output), dim=-1)

        # Get logits from the classifier
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            # Compute loss if labels are provided
            loss = self.loss_fct(logits, labels)

        # Return a dictionary containing loss and logits
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def predict(self, pixel_values, tabular_features, device="cpu"):
        """
        Performs inference on the given input data.

        Args:
            pixel_values (torch.Tensor): Preprocessed image tensor (batch_size, channels, height, width).
            tabular_features (torch.Tensor): Preprocessed tabular features tensor (batch_size, num_tabular_features).
            device (str or torch.device): The device to run inference on ('cpu' or 'cuda').

        Returns:
            tuple: A tuple containing:
                - predicted_class_ids (list): List of predicted class IDs.
                - predicted_probabilities (list): List of class probabilities for the predicted class.
                - all_class_probabilities (list): List of probability distributions over all classes.
        """
        # Set the model to evaluation mode
        self.eval()
        self.to(device)

        # Move inputs to the specified device
        pixel_values = pixel_values.to(device)
        tabular_features = tabular_features.to(device)

        # Perform forward pass
        outputs = self.forward(
            pixel_values=pixel_values, tabular_features=tabular_features
        )
        logits = outputs["logits"]

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get predicted class ID and probability
        predicted_class_probabilities, predicted_class_ids = torch.max(
            probabilities, dim=-1
        )

        return (
            predicted_class_ids.cpu().tolist(),
            predicted_class_probabilities.cpu().tolist(),
            probabilities.cpu().tolist(),
        )

    @torch.no_grad()
    def full_predict(
        self, raw_image: Any, raw_age: int, raw_localization: str, device: str = "cpu"
    ) -> Tuple[str, float]:
        """
        Performs the complete inference pipeline from raw inputs to a human-readable prediction.
        Combines preprocessing, model inference, and postprocessing.

        Args:
            raw_image (Any): The raw image input (e.g., PIL.Image.Image).
            raw_age (int): The raw age of the patient.
            raw_localization (str): The raw localization string.
            device (str): The device to run inference on ('cpu' or 'cuda').

        Returns:
            Tuple[str, float]: A tuple containing:
                - predicted_dx_label (str): The human-readable diagnosis label.
                - predicted_confidence_score (float): The confidence score for the predicted label.
        """
        self.eval()  # Ensure model is in evaluation mode
        self.to(device)  # Move model to target device

        # Image preprocessing
        img_rgb = raw_image.convert("RGB")
        processed_img = self.image_processor(img_rgb, return_tensors="pt")
        pixel_values = processed_img["pixel_values"].to(device)  # Move to device here

        # Tabular features preprocessing
        localization_one_hot = torch.zeros(
            self.config.num_localization_features, device=device
        )
        if raw_localization in self.config.localization_to_id:
            localization_one_hot[self.config.localization_to_id[raw_localization]] = 1.0

        def normalize_age_func_reconstructed(age_value):
            if age_value is None:
                return (self.config.age_mean - self.config.age_min) / (
                    self.config.age_max - self.config.age_min
                )
            return (
                (age_value - self.config.age_min)
                / (self.config.age_max - self.config.age_min)
                if (self.config.age_max - self.config.age_min) > 0
                else 0.0
            )

        age_normalized_value = normalize_age_func_reconstructed(raw_age)
        age_normalized = torch.tensor(
            [age_normalized_value], dtype=torch.float, device=device
        )

        tabular_features = torch.cat([localization_one_hot, age_normalized]).unsqueeze(
            0
        )  # Add batch dimension (1, total_features_dim)

        # Model Inference
        predicted_class_ids_list, predicted_probabilities_list, _ = self.predict(
            pixel_values=pixel_values,
            tabular_features=tabular_features,
            device=device,  # Pass device to predict
        )

        # Since full_predict handles single sample, extract first element
        predicted_class_id = predicted_class_ids_list[0]
        predicted_confidence_score = predicted_probabilities_list[0]

        # Postprocessing
        predicted_dx_label = self.config.id2label[str(predicted_class_id)]

        return predicted_dx_label, predicted_confidence_score
