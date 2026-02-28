from hypso import Hypso
import numpy as np
from enum import Enum
import torch
from multiprocessing import Queue


class DataProduct(Enum):
    L1A = 'l1a'
    L1B = 'l1b'
    L1D = 'l1d'

def producer(file_list : list[str], data_queue : Queue, data_product : DataProduct, batch_size : int):
    """
    Reads data from files and puts slices into the queue for the consumer to process.
    Args: 
    file_list (list[str]): List of file paths to read data from.
    data_queue (Queue): A thread-safe queue to put the data slices into.
    data_product (DataProduct): Enum indicating which data product to read (L1A, L1B, or L1D).
    """
    print(  f"Total files to process: {len(file_list)}"  )

    product_to_attr = {
    DataProduct.L1A: "l1a_cube",
    DataProduct.L1B: "l1b_cube",
    DataProduct.L1D: "l1d_cube"
    }

    cube_attr_name = product_to_attr.get(data_product)
    if not cube_attr_name:
        raise ValueError(f"Unknown data product: {data_product}")

    for file_path in file_list:
        satobj = Hypso(file_path)
        
        cube = getattr(satobj, cube_attr_name, None)
        if cube is None:
            print(f"Skipping {file_path}: Missing {cube_attr_name} data.")
            continue

        pixels = cube.values.astype(np.float32)
        h, w, b = pixels.shape
        pixels_2d = pixels.reshape(-1, b) # Shape: (Total_Pixels_In_Image, 120)

        for i in range(0, pixels_2d.shape[0], batch_size):
            batch = pixels_2d[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            data_queue.put(batch_tensor)
                
    # Send a "done" signal
    data_queue.put("FINISHED")


def fetch_init_data(file_list : list[str], n_target_samples : int, data_product : DataProduct):
    """
    Reads a random subsample of pixels from files and returns the full tensor.
    Args: 
    file_list (list[str]): List of file paths to read data from.
    n_target_samples (int): Number of training-samples
    data_product (DataProduct): Enum indicating which data product to read (L1A, L1B, or L1D).
    Return:

    """

    len_file_list = len(file_list)
    # We read from max 30 files to reduce computation time during development, but this can be adjusted as needed.
    MAX_FILES_TO_READ = 20
    num_files = min(len_file_list, MAX_FILES_TO_READ)
    file_list = file_list[:num_files] 
    samples_per_file = n_target_samples // num_files
    sampled_pixels = []
    print(f"Aiming to extract ~{samples_per_file} pixels per file from {num_files} files to reach a total of ~{n_target_samples} samples.")
    i = 1

    product_to_attr = {
    DataProduct.L1A: "l1a_cube",
    DataProduct.L1B: "l1b_cube",
    DataProduct.L1D: "l1d_cube"
    }

    cube_attr_name = product_to_attr.get(data_product)
    if not cube_attr_name:
        raise ValueError(f"Unknown data product: {data_product}")
    
    for file_path in file_list:

        satobj = Hypso(file_path)

        cube = getattr(satobj, cube_attr_name, None)
        if cube is None:
            print(f"Skipping {file_path}: Missing {cube_attr_name} data.")
            continue

        pixels = cube.values.astype(np.float32)
    
        h, w, b = pixels.shape
        image_2d = pixels.reshape(-1, b) # Shape: (Total_Pixels_In_Image, 120)
        # Random Subsampling 
        total_pixels_in_image = image_2d.shape[0]
        n_to_take = min(samples_per_file, total_pixels_in_image)
        # Generate random indices
        rng = np.random.default_rng()
        indices = rng.choice(total_pixels_in_image, size=n_to_take, replace=False)
        # Grab the random pixels and add to list
        sampled_pixel_subset = image_2d[indices, :]
        sampled_pixels.append(sampled_pixel_subset)
        print(f"{i}/{num_files} | File: {file_path} | Extracted {n_to_take} pixels.")
        i += 1

    data = np.concatenate(sampled_pixels, axis=0)
    print("-" * 30)
    print(f"Final Dataset Shape: {data.shape}")
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    return data_tensor 