import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)


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
