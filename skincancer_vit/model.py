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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model_checkpoint = vision_model_checkpoint
        self.total_tabular_features_dim = total_tabular_features_dim
        self.num_dx_labels = num_dx_labels


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

        # Initialize weights for custom layers (tabular_mlp, classifier)
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
