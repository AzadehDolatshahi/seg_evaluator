# metrics_main.py
import argparse
from pathlib import Path
from utils import load_config, setup_logging, prepare_metrics_dict
from model_utils import load_model, load_preprocess_postprocess
from zip_utils import read_image_from_zip, read_mask_from_zip
from metrics_calculation import calculate_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Metrics on Retina Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

def main():
    # Setup logger and parse arguments
    logger = setup_logging()
    args = parse_arguments()
    config_path = Path(args.config)

    # Load configuration
    config = load_config(config_path)
    
    # Extract paths from the configuration
    try:
        test_dir_path = Path(config['paths']['test_dir'])
        mask_dir_path = Path(config['paths']['mask_dir'])
        model_path = Path(config['paths']['model_path'])
        preprocess_postprocess_path = Path(config['paths']['preprocess_postprocess_path'])
        output_csv_path = Path(config['paths'].get('output_csv', 'metrics_results.csv'))
    except KeyError as e:
        logger.error(f"Missing key in configuration file: {e}")
        exit(1)
    
    # Load preprocess and postprocess functions
    preprocess_function, postprocess_function = load_preprocess_postprocess(preprocess_postprocess_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    # Initialize metrics dictionary
    metrics_dict = prepare_metrics_dict()

    # Run the actual metrics evaluation logic (extract, preprocess, calculate metrics)
    calculate_metrics(test_dir_path, mask_dir_path, model, preprocess_function, postprocess_function, device, metrics_dict)

    # Optionally save the results
    # (Write CSV logic here or in the utils module)
    
if __name__ == '__main__':
    main()
