import os
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

from skincancer_vit.model import SkinCancerViTModel, SkinCancerViTModelConfig
from skincancer_vit.data import (
    CustomDataCollator,
    load_and_prepare_data,
    create_preprocessing_function,
)


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
        normalize_age_func,
        age_mean,
        age_std,
        age_min,
        age_max,
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
        normalize_age_func,
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
    model_config = SkinCancerViTModelConfig(
        vision_model_checkpoint=model_checkpoint,
        total_tabular_features_dim=total_tabular_features_dim,
        num_dx_labels=num_dx_labels,
        id2label=id2label,
        label2id=label2id,
        localization_to_id=localization_to_id,
        num_localization_features=num_localization_features,
        age_mean=float(age_mean),
        age_std=float(age_std),
        age_min=float(age_min),
        age_max=float(age_max),
    )
    model = SkinCancerViTModel(model_config)  # Initialize with the config object
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
