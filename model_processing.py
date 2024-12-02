import torch
from pathlib import Path
import numpy as np
from PIL import Image
import zipfile
from io import BytesIO
import logging
import importlib.util
import sys

def load_preprocess_postprocess(preprocess_postprocess_path: Path):
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
            raise AttributeError("Preprocess or Postprocess functions not found in the module.")
        return preprocess_function, postprocess_function
    except Exception as e:
        logging.error(f"Error loading preprocess/postprocess functions: {e}")
        sys.exit(1)

def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Loads the PyTorch model from the specified path.

    Args:
        model_path (Path): The absolute path to the model file.
        device (torch.device): The device to map the model to.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    logging.info(f"Attempting to load model from: {model_path}")
    if not model_path.exists():
        logging.error(f"Model file not found at {model_path}")
        sys.exit(1)
    
    try:
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        sys.exit(1)

def read_image_from_zip(zip_file: zipfile.ZipFile, file_name: str, color_mode='RGB') -> Image.Image:
    """
    Reads an image from a ZIP file and returns it as a PIL Image.

    Args:
        zip_file (zipfile.ZipFile): The ZIP file object.
        file_name (str): The name of the file to read.
        color_mode (str): The color mode to convert the image to. 'RGB' for color,
                          'L' for grayscale. Default is 'RGB'.

    Returns:
        Image.Image: The loaded image, converted to the specified color mode.
    """
    with zip_file.open(file_name) as file:
        image_data = file.read()
        image = Image.open(BytesIO(image_data))
        return image.convert(color_mode)

def read_mask_from_zip(zip_file: zipfile.ZipFile, file_name: str) -> np.ndarray:
    """
    Reads a mask image from a ZIP file and returns it as a NumPy array.

    Args:
        zip_file (zipfile.ZipFile): The ZIP file object.
        file_name (str): The name of the mask file to read.

    Returns:
        np.ndarray: The loaded mask as a binary NumPy array.
    """
    with zip_file.open(file_name) as file:
        mask_data = file.read()
        mask = Image.open(BytesIO(mask_data)).convert('1')  # Convert to binary
        mask_np = np.array(mask).astype(np.uint8)
        return mask_np

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    # Update these paths as necessary
    preprocess_postprocess_path = Path('/path/to/your/preprocess_postprocess.py')
    model_path = Path('/path/to/your/model.pt')  # Assuming a PyTorch model
    device = torch.device('cpu')  # Use 'cuda' for GPU

    preprocess, postprocess = load_preprocess_postprocess(preprocess_postprocess_path)
    model = load_model(model_path, device)
