import numpy as np
import sys
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score, jaccard_score, accuracy_score
from surface_distance import compute_surface_distances, compute_robust_hausdorff
from zip_utils import read_image_from_zip, read_mask_from_zip
import logging


def dice_coefficient(y_true, y_pred):
    """
    Calculates the Dice Coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    if np.sum(y_true) + np.sum(y_pred) == 0:
        return 1.0  # Both masks are empty
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def calculate_sensitivity_specificity(mask_cls, pred_cls):
    """
    Calculates sensitivity and specificity, handling edge cases where
    TP+FN or TN+FP equals zero.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(mask_cls, pred_cls, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing sensitivity/specificity: {e}")
        sensitivity, specificity = np.nan, np.nan
    return sensitivity, specificity

def calculate_hausdorff(mask_bool, pred_bool):
    """
    Calculates Hausdorff and Hausdorff 95% distance, handling empty masks.
    """
    try:
        if np.any(mask_bool) and np.any(pred_bool):
            surface_distances = compute_surface_distances(mask_bool, pred_bool, [1.0, 1.0])
            hausdorff_full = compute_robust_hausdorff(surface_distances, 100)
            hausdorff_95 = compute_robust_hausdorff(surface_distances, 95)
        else:
            # If one or both masks are empty, return NaN for Hausdorff distances
            hausdorff_full = hausdorff_95 = np.nan
    except Exception as e:
        logging.error(f"Error computing Hausdorff distances: {e}")
        hausdorff_full, hausdorff_95 = np.nan
    return hausdorff_full, hausdorff_95

def calculate_metrics(test_dir, mask_dir, model, preprocess_function, postprocess_function, device, metrics_dict):
    test_image_files = sorted([f for f in test_dir.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    if not test_image_files:
        logging.warning(f"No test images found.")
        return

    for img_file in test_image_files:
        filename = img_file.split('/')[-1]
        mask_file = filename

        if mask_file not in mask_dir.namelist():
            logging.warning(f"Mask file {mask_file} not found, skipping.")
            continue

        # Load image and mask, preprocess them
        try:
            img = read_image_from_zip(test_dir, img_file)
            img_np = np.array(img)
            img_preprocessed = preprocess_function(img_np)
            img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float().to(device)
        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")
            continue

        try:
            mask_np = read_mask_from_zip(mask_dir, mask_file)
        except Exception as e:
            logging.error(f"Error processing mask {filename}: {e}")
            continue

        # Predict and post-process the prediction
        try:
            with torch.no_grad():
                pred = model(img_tensor).cpu().numpy().squeeze()
                pred_postprocessed = postprocess_function(pred)
                pred_binary = pred_postprocessed.astype(np.uint8)
        except Exception as e:
            logging.error(f"Error during prediction for image {filename}: {e}")
            continue

        pred_flat = pred_binary.flatten()
        mask_flat = mask_np.flatten()

        # Calculate metrics for each class
        for cls in [1, 0]:
            pred_cls = (pred_flat == cls).astype(np.uint8)
            mask_cls = (mask_flat == cls).astype(np.uint8)

            # Dice Coefficient
            dice = dice_coefficient(mask_cls, pred_cls)

            # Hausdorff Distance
            mask_bool = mask_cls.reshape(mask_np.shape).astype(bool)
            pred_bool = pred_cls.reshape(mask_np.shape).astype(bool)
            hausdorff_full, hausdorff_95 = calculate_hausdorff(mask_bool, pred_bool)

            # Precision, Recall, F-beta, Jaccard, Accuracy
            try:
                precision = precision_score(mask_cls, pred_cls, zero_division=0)
                recall = recall_score(mask_cls, pred_cls, zero_division=0)
                fbeta_05 = fbeta_score(mask_cls, pred_cls, beta=0.5, zero_division=0)
                fbeta_2 = fbeta_score(mask_cls, pred_cls, beta=2, zero_division=0)
                jaccard = jaccard_score(mask_cls, pred_cls, zero_division=0)
                accuracy = accuracy_score(mask_cls, pred_cls)
            except Exception as e:
                logging.error(f"Error computing classification metrics for image {filename}, class {cls}: {e}")
                precision = recall = fbeta_05 = fbeta_2 = jaccard = accuracy = np.nan

            # Sensitivity and Specificity
            sensitivity, specificity = calculate_sensitivity_specificity(mask_cls, pred_cls)

            # Store Metrics
            metrics_dict['Dice Coefficient'].append((cls, dice))
            metrics_dict['Hausdorff Distance'].append((cls, hausdorff_full))
            metrics_dict['Hausdorff 95% Distance'].append((cls, hausdorff_95))
            metrics_dict['Precision'].append((cls, precision))
            metrics_dict['Recall'].append((cls, recall))
            metrics_dict['F-beta (beta=0.5)'].append((cls, fbeta_05))
            metrics_dict['F-beta (beta=2)'].append((cls, fbeta_2))
            metrics_dict['Jaccard Index (IoU)'].append((cls, jaccard))
            metrics_dict['Accuracy'].append((cls, accuracy))
            metrics_dict['Sensitivity'].append((cls, sensitivity))
            metrics_dict['Specificity'].append((cls, specificity))

        logging.info(f"Processed image: {filename}")


    # Create DataFrame for Displaying Metrics
    df = pd.DataFrame(columns=['Overall', 'Object Class', 'Background Class'], index=list(metrics_dict.keys()))
    for metric in metrics_dict.keys():
        object_scores = [score for cls, score in metrics_dict[metric] if cls == 1]
        background_scores = [score for cls, score in metrics_dict[metric] if cls == 0]

        overall_score = np.nanmean(object_scores + background_scores) if (object_scores + background_scores) else np.nan
        object_mean = np.nanmean(object_scores) if object_scores else np.nan
        background_mean = np.nanmean(background_scores) if background_scores else np.nan

        df.loc[metric] = [overall_score, object_mean, background_mean]

    # Print the DataFrame in Table Format
    try:
        import tabulate
        print(df.to_markdown(tablefmt="grid"))
    except ImportError:
        logging.warning("tabulate module not found. Printing DataFrame using default format.")
        print(df)

    return df
