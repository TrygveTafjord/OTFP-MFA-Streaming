from hypso import Hypso
import numpy as np
from enum import Enum
import torch
from multiprocessing import Queue


class DataProduct(Enum):
    L1A = 'l1a'
    L1B = 'l1b'
    L1D = 'l1d'

def producer(file_list : list[str], data_queue : Queue, data_product : DataProduct, batch_frames : int):
    """
    Reads data from files and puts slices into the queue for the consumer to process.
    """
    print(f"Total files to process: {len(file_list)}")

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
        if cube is None: continue

        # pixels is a NumPy array of shape (h, w, d)
        pixels = cube.values.astype(np.float32) 
        h, w, d = pixels.shape

        # Loop through the image in chunks of `batch_frames`
        for i in range(0, h, batch_frames):
            
            # 1. Grab the 3D batch: shape (batch_frames, w, d)
            # Note: The last batch might be smaller than batch_frames, which is fine!
            batch_3d = pixels[i : i + batch_frames, :, :]
            
            # 2. Get the actual number of lines in this specific chunk
            current_batch_lines = batch_3d.shape[0]
            
            # 3. Use NumPy's .reshape() to flatten the spatial dimensions!
            # Shape becomes (current_batch_lines * w, d)
            projection_data_flat = batch_3d.reshape(current_batch_lines * w, d)
            
            # 4. Put the flattened 2D batch into the queue
            data_queue.put(projection_data_flat)
                
    data_queue.put("FINISHED")

