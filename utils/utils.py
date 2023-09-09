from typing import List, Optional

from PIL import Image
import numpy as np
import torch
import safetensors.torch

import utils.checkpoint_pickle


"""
Original code from ComfyUI - https://github.com/comfyanonymous/ComfyUI
"""
def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=utils.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def make_grid(images: List[Image.Image], rows: Optional[int] = None, cols: Optional[int] = None):
    # Get max image size
    max_width = 0
    max_height = 0
    for image in images:
        max_width = max(max_width, image.width)
        max_height = max(max_height, image.height)

    if rows is None and cols is None:
        # Create square grid by default
        rows = cols = int(np.ceil(len(images) ** 0.5))
    elif rows is None:
        rows = int(np.ceil(len(images) / cols))
    elif cols is None:
        cols = int(np.ceil(len(images) / rows))
    elif rows * cols < len(images):
        print(f"Insufficient cells in { rows } x { cols } grid, only showing first { rows * cols } images")
        images = images[:rows * cols]
    
    grid = Image.new("RGB", (cols * max_width, rows * max_height))
    for i, image in enumerate(images):
        grid.paste(image, (i % cols * max_width, i // cols * max_height))
    return grid
