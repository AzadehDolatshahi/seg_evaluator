import numpy as np
import sys
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from surface_distance import compute_surface_distances, compute_robust_hausdorff
from zip_utils import read_image_from_zip, read_mask_from_zip
import logging
import torch 
import yaml


def calculate_dsc_aggregated(total_tp, total_fp, total_fn):  # calculate_dsc_aggregated
    """
    Calculates the aggregated Dice Coefficient (DSC) using the formula:
    DSC = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        total_tp (int): Total true positives across the dataset.
        total_fp (int): Total false positives across the dataset.
        total_fn (int): Total false negatives across the dataset.
        
    Returns:
        dsc (float): The aggregated Dice Coefficient (DSC).
    """
    try:
        denominator = (2 * total_tp) + total_fp + total_fn
        if denominator != 0:
            dsc = (2 * total_tp) / denominator
        else:
            dsc = np.nan
    except Exception as e:
        logging.error(f"Error computing aggregated DSC: {e}")
        dsc = np.nan

    return dsc



def calculate_fbeta_iou_aggregated(total_tp, total_fp, total_fn, beta=1.0):
    """
    Calculates F-beta score and IoU using aggregated TP, FP, and FN based on the provided formula.
    """
    try:
        beta_squared = beta ** 2

        # F-beta score using the formula
        if total_tp + total_fp + beta_squared * total_fn != 0:
            fbeta = (1 + beta_squared) * total_tp / ((1 + beta_squared) * total_tp + total_fp + beta_squared * total_fn)
        else:
            fbeta = np.nan

        # IoU
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing F-beta and IoU: {e}")
        fbeta, iou = np.nan, np.nan

    return fbeta, iou


def calculate_precision_recall_accuracy_aggregated(total_tp, total_tn, total_fp, total_fn):
    """
    Calculates precision, recall, and accuracy using aggregated TP, TN, FP, and FN.
    """
    try:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else np.nan
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else np.nan
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing precision/recall/accuracy: {e}")
        precision, recall, accuracy = np.nan, np.nan, np.nan
    return precision, recall, accuracy


def calculate_sensitivity_specificity_aggregated(total_tp, total_tn, total_fp, total_fn):
    """
    Calculates sensitivity and specificity from aggregated TP, TN, FP, and FN.
    """
    try:
        sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else np.nan
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else np.nan
    except Exception as e:
        logging.error(f"Error computing aggregated sensitivity/specificity: {e}")
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

    # Initialize accumulators for confusion matrix elements
    total_tp = {0: 0, 1: 0}
    total_tn = {0: 0, 1: 0}
    total_fp = {0: 0, 1: 0}
    total_fn = {0: 0, 1: 0}

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
            #dice = calculate_dsc_aggregated(mask_cls, pred_cls) #(total_tp, total_fp, total_fn)

            # Hausdorff Distance
            mask_bool = mask_cls.reshape(mask_np.shape).astype(bool)
            pred_bool = pred_cls.reshape(mask_np.shape).astype(bool)
            hausdorff_full, hausdorff_95 = calculate_hausdorff(mask_bool, pred_bool)

            # Confusion matrix values
            try:
                tn, fp, fn, tp = confusion_matrix(mask_cls, pred_cls, labels=[0, 1]).ravel()
                total_tp[cls] += tp
                total_tn[cls] += tn
                total_fp[cls] += fp
                total_fn[cls] += fn
            except Exception as e:
                logging.error(f"Error computing confusion matrix for image {filename}, class {cls}: {e}")
                continue

            # Store Dice and Hausdorff metrics
            #metrics_dict['Dice Coefficient'].append((cls, dice))
            metrics_dict['Hausdorff Distance'].append((cls, hausdorff_full))
            metrics_dict['Hausdorff 95% Distance'].append((cls, hausdorff_95))

        logging.info(f"Processed image: {filename}")

    # Calculate aggregated sensitivity, specificity, precision, recall, accuracy, F-beta, and IoU
    for cls in [1, 0]:
        sensitivity, specificity = calculate_sensitivity_specificity_aggregated(total_tp[cls], total_tn[cls], total_fp[cls], total_fn[cls])
        precision, recall, accuracy = calculate_precision_recall_accuracy_aggregated(total_tp[cls], total_tn[cls], total_fp[cls], total_fn[cls])
        fbeta_05, iou = calculate_fbeta_iou_aggregated(total_tp[cls], total_fp[cls], total_fn[cls], beta=0.5)
        fbeta_2, _ = calculate_fbeta_iou_aggregated(total_tp[cls], total_fp[cls], total_fn[cls], beta=2)
        dsc = calculate_dsc_aggregated(total_tp[cls], total_fp[cls], total_fn[cls])


        metrics_dict['Sensitivity'].append((cls, sensitivity))
        metrics_dict['Specificity'].append((cls, specificity))
        metrics_dict['Precision'].append((cls, precision))
        metrics_dict['Recall'].append((cls, recall))
        metrics_dict['Accuracy'].append((cls, accuracy))
        metrics_dict['F-beta (beta=0.5)'].append((cls, fbeta_05))
        metrics_dict['F-beta (beta=2)'].append((cls, fbeta_2))
        metrics_dict['Jaccard Index (IoU)'].append((cls, iou))
        metrics_dict['Dice Coefficient'].append((cls, dsc))

    # Create DataFrame for Displaying Metrics
    df = pd.DataFrame(columns=['Overall', 'Object Class', 'Background Class'], index=list(metrics_dict.keys()))
    for metric in metrics_dict.keys():
        object_scores = [score for cls, score in metrics_dict[metric] if cls == 1]
        background_scores = [score for cls, score in metrics_dict[metric] if cls == 0]

        overall_score = np.nanmean(object_scores + background_scores) if (object_scores + background_scores) else np.nan
        object_mean = np.nanmean(object_scores) if object_scores else np.nan
        background_mean = np.nanmean(background_scores) if background_scores else np.nan

        df.loc[metric] = [overall_score, object_mean, background_mean]

    # Ensure all values in the DataFrame are numeric (or NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    # Round DataFrame values to 4 decimal places
    df = df.round(4)
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    save_path = config['paths']['output_csv']
    csv_file_path = f"{save_path}"
    df.to_csv(csv_file_path, index=False)


    # Print the DataFrame 
    try:
        import tabulate
        print(df.to_markdown(tablefmt="grid"))
    except ImportError:
        logging.warning("tabulate module not found. Printing DataFrame using default format.")
        print(df)

