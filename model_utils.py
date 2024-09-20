import torch
import importlib.util
import logging
import sys

def load_preprocess_postprocess(preprocess_postprocess_path):
    """
    Dynamically loads preprocess and postprocess functions from a specified module.
    """
    try:
        spec = importlib.util.spec_from_file_location("preprocess_postprocess_module", preprocess_postprocess_path)
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)
        preprocess_function = getattr(preprocess_module, 'preprocess_function', None)
        postprocess_function = getattr(preprocess_module, 'postprocess_function', None)
        if not preprocess_function or not postprocess_function:
            logging.error("Preprocess or Postprocess functions not found.")
            sys.exit(1)
        return preprocess_function, postprocess_function
    except Exception as e:
        logging.error(f"Error loading preprocess/postprocess functions: {e}")
        sys.exit(1)

def load_model(model_path, device):
    """
    Loads the PyTorch model from the specified path.
    """
    try:
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        sys.exit(1)
