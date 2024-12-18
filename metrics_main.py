#!/usr/bin/env python3
# to Run python metrics_main.py --config config.yaml
import argparse
from pathlib import Path
from utils import load_config, setup_logging, prepare_metrics_dict
from model_utils import load_model, load_preprocess_postprocess
from zip_utils import read_image_from_zip, read_mask_from_zip
from metrics_calculation import calculate_metrics
import zipfile
import torch

def parse_arguments():
    """
     command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run Metrics on Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

def main():

    logger = setup_logging()
    args = parse_arguments()
    config_path = Path(args.config)


    config = load_config(config_path)
    
    # Extract paths from the configuration
    try:
        test_dir_path = Path(config['paths']['test_dir'])
        mask_dir_path = Path(config['paths']['mask_dir'])
        model_path = Path(config['paths']['model_path'])
        
        # Check if preprocess_postprocess_path is provided; if not, set to None
        preprocess_postprocess_path_str = config['paths'].get('preprocess_postprocess_path', None)
        if preprocess_postprocess_path_str is not None:
            preprocess_postprocess_path = Path(preprocess_postprocess_path_str)
        else:
            preprocess_postprocess_path = None
        
    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        exit(1)

    # 
    if preprocess_postprocess_path is not None:
        preprocess, postprocess = load_preprocess_postprocess(preprocess_postprocess_path)
    else:
        # handle the lack of preprocessing/postprocessing
        preprocess, postprocess = None, None


    # Load preprocess and postprocess functions
    preprocess_function, postprocess_function = load_preprocess_postprocess(preprocess_postprocess_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    # Initialize metrics dictionary
    metrics_dict = prepare_metrics_dict()

    #  ZIP files
    try:
        test_dir = zipfile.ZipFile(test_dir_path, 'r')
        mask_dir = zipfile.ZipFile(mask_dir_path, 'r')
    except zipfile.BadZipFile as e:
        logger.error(f"Error opening ZIP files: {e}")
        exit(1)


    calculate_metrics(test_dir, mask_dir, model, preprocess_function, postprocess_function, device, metrics_dict)

    #  save the results here later

if __name__ == '__main__':
    main()
