import os
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
)
import numpy as np
import evaluate
from PIL import Image
from collections import defaultdict


class CombinedModelConfig(PretrainedConfig):
    model_type = (
        "combined_multimodal_skin_cancer"  # A unique model type for your custom model
    )

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


class CombinedModel(PreTrainedModel):
    config_class = CombinedModelConfig  # Link to your custom config class

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


class CustomDataCollator:
    def __call__(self, features):
        pixel_values_to_stack = []
        tabular_features_to_stack = []
        labels_to_stack = []

        for f in features:
            # Keep unwrapping if it's a list and contains elements
            # This unwrapping logic is kept for safety, but with set_format, pixel_values should ideally already be tensors.
            px_val = f["pixel_values"]
            while isinstance(px_val, list) and len(px_val) > 0:
                px_val = px_val[0]  # Get the first element if it's a list

            pixel_values_to_stack.append(px_val)

            # Apply the same logic
            tab_val = f["tabular_features"]
            while isinstance(tab_val, list) and len(tab_val) > 0:
                tab_val = tab_val[0]

            tabular_features_to_stack.append(tab_val)

            # Labels are typically integers
            labels_to_stack.append(f["labels"])

        pixel_values = torch.stack(pixel_values_to_stack)
        tabular_features = torch.stack(tabular_features_to_stack)
        labels = torch.tensor(labels_to_stack)

        # Return a dictionary formatted for the Trainer
        return {
            "pixel_values": pixel_values,
            "tabular_features": tabular_features,
            "labels": labels,
        }


def load_and_prepare_data(num_records_to_use=1000):
    """
    Loads the skin cancer dataset, shuffles it, selects a subset of records,
    defines diagnosis labels, and pre-computes mappings/normalization for tabular features.
    """
    print("Loading dataset 'marmal88/skin_cancer'...")
    dataset = load_dataset("marmal88/skin_cancer")
    print("Dataset loaded successfully.")

    # Shuffle and select a subset of records for each split
    print(f"Shuffling and selecting {num_records_to_use} records from each split...")
    for split_name in dataset.keys():
        dataset[split_name] = (
            dataset[split_name]
            .shuffle(seed=42)
            .select(range(min(num_records_to_use, len(dataset[split_name]))))
        )
    print("Dataset subset created.")
    print(dataset)

    # Define Labels and Mappings for 'dx' (Diagnosis)
    # Collect all unique 'dx' values from the 'train' split to define labels
    unique_dx_labels = set()
    for example in dataset["train"]:
        unique_dx_labels.add(example["dx"])
    labels = sorted(list(unique_dx_labels))  # Sort to ensure consistent ID assignment

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    num_dx_labels = len(labels)
    print(f"Number of diagnosis classes (dx): {num_dx_labels}")
    print(f"Diagnosis Labels: {labels}")

    # Pre-compute Mappings and Normalization for Tabular Features ('age' and 'localization')
    all_localizations = set()
    all_ages = []

    for split in dataset:
        for example in dataset[split]:
            all_localizations.add(example["localization"])
            if example["age"] is not None:
                all_ages.append(example["age"])

    localization_names = sorted(list(all_localizations))
    localization_to_id = {name: i for i, name in enumerate(localization_names)}

    min_age = min(all_ages) if all_ages else 0
    max_age = (
        max(all_ages) if all_ages else 100
    )  # Default if no ages to prevent division by zero

    def normalize_age(age):
        if age is None:
            # Handle missing age: for simplicity, use 0.0 (normalized value)
            # This corresponds to the min_age if min_age == max_age, or a normalized average.
            if max_age == min_age:  # Avoid division by zero if all ages are the same
                return 0.0
            return (np.mean(all_ages) - min_age) / (max_age - min_age)
        return (age - min_age) / (max_age - min_age) if (max_age - min_age) > 0 else 0.0

    num_localization_features = len(localization_names)
    num_age_features = 1  # For normalized age

    total_tabular_features_dim = num_localization_features + num_age_features
    print(
        f"Total tabular features dimension (localization + age): {total_tabular_features_dim}"
    )

    return (
        dataset,
        label2id,
        id2label,
        num_dx_labels,
        localization_to_id,
        num_localization_features,
        normalize_age,
        total_tabular_features_dim,
    )


def create_preprocessing_function(
    image_processor,
    label2id,
    localization_to_id,
    num_localization_features,
    normalize_age,
):
    """
    Creates and returns the preprocessing function for multimodal data.
    This function now processes a SINGLE example.
    """

    def preprocess_example_multimodal(example):  # Takes a single example
        # Image preprocessing
        img_rgb = example["image"].convert("RGB")

        # Process the single image. This should return a BatchFeature.
        processed_img = image_processor(img_rgb, return_tensors="pt")

        # Extract pixel_values. It should be a tensor (1, C, H, W).
        pixel_values_tensor = processed_img["pixel_values"]

        # Explicitly ensure it's a tensor and squeeze the batch dimension.
        # This is the crucial part to handle cases where image_processor might not return a tensor directly.
        if not isinstance(pixel_values_tensor, torch.Tensor):
            # Attempt to convert from numpy array or list to tensor
            try:
                pixel_values_tensor = torch.tensor(
                    pixel_values_tensor, dtype=torch.float32
                )
            except Exception as e:
                raise TypeError(
                    f"Could not convert pixel_values to torch.Tensor. Original type: {type(pixel_values_tensor)}, Error: {e}"
                )

        pixel_values = pixel_values_tensor.squeeze(
            0
        )  # Remove the batch dimension (1, C, H, W) -> (C, H, W)

        # Tabular feature processing (for this single example)
        localization_one_hot = torch.zeros(num_localization_features)
        if example["localization"] in localization_to_id:
            localization_one_hot[localization_to_id[example["localization"]]] = 1.0

        age_normalized = torch.tensor(
            [normalize_age(example["age"])], dtype=torch.float
        )

        tabular_features = torch.cat([localization_one_hot, age_normalized])

        return {
            "pixel_values": pixel_values,  # This should now definitively be a (C, H, W) tensor
            "tabular_features": tabular_features,  # This is now a single tensor
            "labels": label2id[example["dx"]],  # This is now a single integer label
        }

    return preprocess_example_multimodal


def define_metrics():
    """
    Defines and returns the compute_metrics function for evaluation.
    """
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Get the predicted class IDs by finding the argmax of logits
        predictions = np.argmax(predictions, axis=1)
        # Compute accuracy using the loaded metric
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def setup_training_arguments():
    """
    Configures and returns the TrainingArguments for the Trainer.
    """
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",  # Directory to save checkpoints and logs
        per_device_train_batch_size=16,  # Batch size per GPU/CPU for training
        per_device_eval_batch_size=16,  # Batch size per GPU/CPU for evaluation
        num_train_epochs=3,  # Number of training epochs
        logging_dir="./skin_cancer_multimodal_logs",  # Directory for TensorBoard logs
        logging_steps=50,  # Log training progress every 50 steps
        eval_strategy="epoch",
        save_strategy="epoch",  # Save model checkpoint at the end of each epoch
        load_best_model_at_end=True,  # Load the best model based on evaluation metric at the end of training
        metric_for_best_model="accuracy",  # Metric to monitor for best model
        report_to="none",  # No reporting to external services (e.g., "tensorboard", "wandb")
        push_to_hub=False,  # Set to True if you want to push the model to Hugging Face Hub
    )
    return training_args


def train_and_evaluate_model(
    model, training_args, processed_dataset, compute_metrics, data_collator
):
    """
    Initializes and runs the Trainer, then evaluates the model on the test set.
    """
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,  # Use custom data collator for multimodal inputs
    )
    print("Trainer initialized. Starting training...")

    # Start training
    trainer.train()

    # Evaluate the Model on the Test Set
    print("\nEvaluating model on the test set...")
    test_results = trainer.evaluate(processed_dataset["test"])
    print(f"Test results: {test_results}")

    # Save final model
    final_model_save_path = "./results/final_model"
    os.makedirs(final_model_save_path, exist_ok=True)
    trainer.save_model(final_model_save_path)
    print(f"Final model saved to: {final_model_save_path}")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    # Pass num_records_to_use to load_and_prepare_data to limit the dataset size
    (
        dataset,
        label2id,
        id2label,
        num_dx_labels,
        localization_to_id,
        num_localization_features,
        normalize_age,
        total_tabular_features_dim,
    ) = load_and_prepare_data(num_records_to_use=10000)

    # Choose Model Checkpoint and Load Image Processor
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    print("Image processor loaded.")

    # Create Preprocessing Function
    preprocess_function = create_preprocessing_function(
        image_processor,
        label2id,
        localization_to_id,
        num_localization_features,
        normalize_age,
    )
    print("Preprocessing dataset splits...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=False,  # Process one example at a time
        remove_columns=["image", "dx", "dx_type", "age", "sex", "localization"],
    )
    # Set the format of the dataset to PyTorch tensors
    processed_dataset.set_format(
        type="torch", columns=["pixel_values", "tabular_features", "labels"]
    )
    print("Preprocessing complete.")
    print(processed_dataset)

    # Define and Initialize Custom Multimodal Model
    model_config = CombinedModelConfig(
        vision_model_checkpoint=model_checkpoint,
        total_tabular_features_dim=total_tabular_features_dim,
        num_dx_labels=num_dx_labels,
    )
    model = CombinedModel(model_config)  # Initialize with the config object
    print("Multimodal model initialized.")

    # Create Custom Data Collator
    data_collator = CustomDataCollator()

    # Define Metrics
    compute_metrics = define_metrics()

    # Configure Training Arguments
    training_args = setup_training_arguments()

    # Train and Evaluate Model
    train_and_evaluate_model(
        model, training_args, processed_dataset, compute_metrics, data_collator
    )
