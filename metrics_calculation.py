import yaml
import numpy as np
import sys
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from surface_distance import compute_surface_distances, compute_robust_hausdorff
from zip_utils import read_image_from_zip, read_mask_from_zip
import logging


def dice_coefficient(total_tp, total_fp, total_fn):

        
        # (2 * TP + FP + FN)
    denominator = 2 * total_tp + total_fp + total_fn
        
        # Handle edge case: If no predictions or ground truth exist for this class
    if denominator == 0:
        dice_score= 1.0  # Dice is 1.0 for empty predictions and ground truth
    else:
        dice_score = (2 * total_tp) / denominator
    
    return dice_score



def calculate_fbeta_aggregated(total_tp, total_fp, total_fn, beta=1.0):
    """
    Calculates F-beta score using aggregated TP, FP, and FN 
    """
    try:
        beta_squared = beta ** 2

        if total_tp + total_fp + beta_squared * total_fn != 0:
            fbeta = (1 + beta_squared) * total_tp / ((1 + beta_squared) * total_tp + total_fp + beta_squared * total_fn)
        else:
            fbeta = np.nan
    except Exception as e:
        logging.error(f"Error computing F-beta: {e}")
        fbeta = np.nan
    return fbeta

def calculate_iou_aggregated(total_tp, total_fp, total_fn):
    try:    
        # IoU
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing IoU: {e}")
        iou = np.nan

    return iou



def calculate_precision_sensitivity_specificity_accuracy_aggregated(total_tp, total_tn, total_fp, total_fn):
    """
    Calculates using aggregated TP, TN, FP, and FN.
    """
    try:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else np.nan
        sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else np.nan
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) != 0 else np.nan
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing precision/sensitivity/accuracy: {e}")
        precision, sensitivity, accuracy = np.nan, np.nan, np.nan
    return precision, sensitivity, accuracy, specificity



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
    test_image_files = sorted([f for f in test_dir.namelist() 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not test_image_files:
        logging.warning(f"No test images found.")
        return

    #  all classes from the mask files 
    all_classes = set()
    for img_file in test_image_files:
        mask_file = img_file.split('/')[-1]
        if mask_file in mask_dir.namelist():
            try:
                mask_np = read_mask_from_zip(mask_dir, mask_file)
                unique_labels = np.unique(mask_np)
                all_classes.update(unique_labels)
            except Exception as e:
                logging.error(f"Error reading mask {mask_file}: {e}")
                continue

    # Sort classes so output is consistent
    all_classes = sorted(all_classes)
    logging.info(f"Detected classes: {all_classes}")

    # initialize accumulators for confusion matrix elements 
    total_tp = {cls: 0 for cls in all_classes}
    total_tn = {cls: 0 for cls in all_classes}
    total_fp = {cls: 0 for cls in all_classes}
    total_fn = {cls: 0 for cls in all_classes}

    # process each test image and compute confusion matrix, hausdorff,
    for img_file in test_image_files:
        filename = img_file.split('/')[-1]
        mask_file = filename

        if mask_file not in mask_dir.namelist():
            logging.warning(f"Mask file {mask_file} not found, skipping.")
            continue

        # Read and preprocess image
        try:
            img = read_image_from_zip(test_dir, img_file)
            img_np = np.array(img)
            img_preprocessed = preprocess_function(img_np)
            img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float().to(device)
        except Exception as e:
            logging.error(f"Error processing image {filename}: {e}")
            continue

        # Read mask
        try:
            mask_np = read_mask_from_zip(mask_dir, mask_file)
        except Exception as e:
            logging.error(f"Error processing mask {filename}: {e}")
            continue

        # Predict and postprocess
        try:
            with torch.no_grad():
                pred = model(img_tensor).cpu().numpy().squeeze()

                # Handle model output shape
                if pred.ndim == 2:  # shape (H, W)
                    pred = pred[np.newaxis, :, :]  # -> (1, H, W)
                if pred.ndim == 4:  # shape (N, C, H, W)
                    pred = pred.squeeze(0)         # -> (C, H, W)

                pred_postprocessed = postprocess_function(pred)
                pred_binary = pred_postprocessed.astype(np.uint8)
        except Exception as e:
            logging.error(f"Error during prediction for image {filename}: {e}")
            continue

        # Flatten predictions and mask
        pred_flat = pred_binary.flatten()
        mask_flat = mask_np.flatten()

        # compute metrics for each class 
        for cls in all_classes:
            pred_cls = (pred_flat == cls).astype(np.uint8)
            mask_cls = (mask_flat == cls).astype(np.uint8)

            # Hausdorff Distance
            mask_bool = mask_cls.reshape(mask_np.shape).astype(bool)
            pred_bool = pred_cls.reshape(mask_np.shape).astype(bool)
            hausdorff_full, hausdorff_95 = calculate_hausdorff(mask_bool, pred_bool)

            # Confusion matrix values
            try:
                tn, fp, fn, tp = confusion_matrix(
                    mask_cls, pred_cls, labels=[0, 1]
                ).ravel()
                total_tp[cls] += tp
                total_tn[cls] += tn
                total_fp[cls] += fp
                total_fn[cls] += fn
            except Exception as e:
                logging.error(
                    f"Error computing confusion matrix for image {filename}, class {cls}: {e}"
                )
                continue

            metrics_dict['Hausdorff Distance'].append((cls, hausdorff_full))
            metrics_dict['Hausdorff 95% Distance'].append((cls, hausdorff_95))

        logging.info(f"Processed image: {filename}")

    # aggregate metrics at the end for each class
    for cls in all_classes:
        precision, sensitivity, accuracy, specificity = calculate_precision_sensitivity_specificity_accuracy_aggregated(
            total_tp[cls], total_tn[cls], total_fp[cls], total_fn[cls]
        )
        fbeta_05 = calculate_fbeta_aggregated(total_tp[cls], total_fp[cls], total_fn[cls], beta=0.5)
        fbeta_2  = calculate_fbeta_aggregated(total_tp[cls], total_fp[cls], total_fn[cls], beta=2)
        iou      = calculate_iou_aggregated(total_tp[cls], total_fp[cls], total_fn[cls])
        dice     = dice_coefficient(total_tp[cls], total_fp[cls], total_fn[cls])

        metrics_dict['Sensitivity'].append((cls, sensitivity))
        metrics_dict['Specificity'].append((cls, specificity))
        metrics_dict['Precision'].append((cls, precision))
        metrics_dict['Accuracy'].append((cls, accuracy))
        metrics_dict['F-beta (beta=0.5)'].append((cls, fbeta_05))
        metrics_dict['F-beta (beta=2)'].append((cls, fbeta_2))
        metrics_dict['Jaccard Index (IoU)'].append((cls, iou))
        metrics_dict['Dice Coefficient'].append((cls, dice))

    # dataFrame with columns for each class plus an 'Overall' column ---
    df_cols = ['Overall'] + [f"Class {cls}" for cls in all_classes]
    df = pd.DataFrame(columns=df_cols, index=list(metrics_dict.keys()))

    for metric in metrics_dict.keys():
        # Extract scores for each class
        class_scores_dict = {cls: [] for cls in all_classes}
        for (cls_label, score) in metrics_dict[metric]:
            class_scores_dict[cls_label].append(score)

        # Overall = mean across all classes
        all_scores = []
        for cls_label in all_classes:
            all_scores.extend(class_scores_dict[cls_label])
        overall_score = np.nanmean(all_scores) if all_scores else np.nan

        # Mean for each class
        row_values = [overall_score]
        for cls_label in all_classes:
            scores_for_cls = class_scores_dict[cls_label]
            row_values.append(np.nanmean(scores_for_cls) if scores_for_cls else np.nan)

        df.loc[metric] = row_values

    # Ensure all values in the DataFrame are numeric (or NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    # Round DataFrame values to 4 decimal places
    df = df.round(4)

    # Read config and save CSV
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    save_path = config['paths']['output_csv']
    csv_file_path = f"{save_path}"
    df.to_csv(csv_file_path, index=False)

    # Print DataFrame in table format if tabulate is available
    try:
        import tabulate
        print(df.to_markdown(tablefmt="grid"))
    except ImportError:
        logging.warning("tabulate module not found. Printing DataFrame using default format.")
        print(df)
