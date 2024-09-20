from PIL import Image
import zipfile
import numpy as np
from io import BytesIO

def read_image_from_zip(zip_file, file_name):
    """
    Reads an image from a ZIP file and returns it as a PIL Image.
    """
    with zip_file.open(file_name) as file:
        image_data = file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image

def read_mask_from_zip(zip_file, file_name):
    """
    Reads a mask image from a ZIP file and returns it as a NumPy array.
    """
    with zip_file.open(file_name) as file:
        mask_data = file.read()
        mask = Image.open(BytesIO(mask_data)).convert('1')  # Convert to binary
        mask_np = np.array(mask).astype(np.uint8)
        return mask_np
