import torch
import importlib.util
import logging
import sys
from tensorflow.keras.models import load_model as load_keras_model
from pathlib import Path

# keras to be added 
def load_preprocess_postprocess(preprocess_postprocess_path):
    """
    Dynamically loads preprocess and postprocess functions from a specified module,
    allowing them to be optional.
    """
    if preprocess_postprocess_path is None:
        return None, None  # Return None if no path is provided

    try:
        spec = importlib.util.spec_from_file_location("preprocess_postprocess_module", preprocess_postprocess_path)
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)
        preprocess_function = getattr(preprocess_module, 'preprocess_function', None)
        postprocess_function = getattr(preprocess_module, 'postprocess_function', None)
        return preprocess_function, postprocess_function
    except Exception as e:
        logging.error(f"Error loading preprocess/postprocess functions: {e}")
        sys.exit(1)

def load_model(model_path, device='cpu'):
    # keras to be added
    """
    Loads a model from the specified path. It determines the type of the model 
    based on the file extension and loads it appropriately.
    """
    try:
        if isinstance(model_path, Path):
            model_path = str(model_path)

        if model_path.endswith('.h5') or model_path.endswith('.keras'):
            # Load a Keras model
            model = load_keras_model(model_path)
            return model
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            # Load a PyTorch model
            try:
               
                model = torch.jit.load(model_path, map_location=device)
                model.eval()
                return model
            except RuntimeError:
                model = torch.load(model_path, map_location=device)
                model.eval()
                model.to(device)
                return model
        else:
            raise ValueError("Unsupported model file format. Use '.h5', '.keras' for Keras or '.pt', '.pth' for PyTorch")
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        sys.exit(1)

