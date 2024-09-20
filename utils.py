import yaml
import logging
import sys

def load_config(config_path):
    """
    Loads configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)

def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def prepare_metrics_dict():
    """
    Initializes a dictionary to store metrics.
    """
    return {
        'Dice Coefficient': [],
        'Hausdorff Distance': [],
        'Hausdorff 95% Distance': [],
        'Precision': [],
        'Recall': [],
        'F-beta (beta=0.5)': [],
        'F-beta (beta=2)': [],
        'Jaccard Index (IoU)': [],
        'Accuracy': [],
        'Sensitivity': [],
        'Specificity': []
    }
