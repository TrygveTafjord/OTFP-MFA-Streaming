from ..otfp import MFA_OTFP
from hypso import Hypso
import numpy as np
import torch

"""
Memory Usage Benchmarking Script. 

Goal: 
Measure the memory usage of the MFA_OTFP model during training and inference, for a countineous stream of data. 

Data: 
Use a single image, but periodically add a new signal to the data to simulate a changing environment. This should force the model to spawn new components at regular intervals.
Image used from hypso catalog: HYPSO-1_HSI_20220823T102645Z-l1a.nc, it is from the Trondheim fjord, and contains a mix of land and water, with some clouds.

Test: 
1. Train the MFA_OTFP model on the first N rows of the image.
2. In a loop, feed the model new batches of data, one batch at a time, and measure the memory usage.
3. Periodically (e.g., every M batches), introduce a new signal (e.g., a bright pixel cluster) to the incoming data to trigger component spawning.
4. Stream for a high number of total samples (e.g., 100,000,000), and log: the memory usage, at what batch a new component is spawned

Plot:
X-axis: Total number of frames seen (this allso acts as time in a way)
Y-axis: Memory usage (in GB)
Annotate the plot with vertical lines indicating when new components were spawned, along with the total number of components at that time.
"""

data_path = "../data/memory_usage_benchmarking/trondheim_2022-08-23T10-26-45Z-l1b.nc"

try:
    satobj = Hypso(data_path)
    cube = getattr(satobj, "l1b_cube", None)
    if cube is None:
        raise ValueError("Missing 'l1b_cube' data.")
    
    img_full = cube.values.astype(np.float32)
    print(f"Successfully loaded | Shape: {img_full.shape}")
except Exception as e:
    print(f"FATAL: Error processing {data_path}: {e}")
    raise e

BATCH_SIZE = 5
N_TRAIN_ROWS = 20
L2_NORMALIZATION = True

MFA_OTFP_model=MFA_OTFP(
    n_channels=120,
    device="cpu",
    outlier_update_treshold=1000,
    q_max=5,
    L2_normalization=L2_NORMALIZATION
)

MFA_OTFP_model.fit(torch.from_numpy(img_full[:N_TRAIN_ROWS, :, :].reshape(-1, 120)).float())
