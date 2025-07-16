import torch
import numpy as np

from datasets import load_dataset, DatasetDict


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
