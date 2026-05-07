import os
import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import sys

# Make sure these imports work in your local environment
# 1. Get the absolute path of the parent directory (OTFP-MFA-Streaming)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add it to sys.path so Python knows where to look for modules
sys.path.insert(0, parent_dir)

from otfp import MFA_OTFP
from hypso import Hypso


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

# 1. SETUP AND HYPERPARAMETERS
data_path = os.path.join(parent_dir, "data", "memory_usage_benchmarking", "trondheim_2022-08-23T10-26-45Z-l1b.nc")

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

h, w, num_bands = img_full.shape
pixels_per_row = w

BATCH_ROWS = 5 
PIXELS_PER_BATCH = BATCH_ROWS * pixels_per_row
N_TRAIN_ROWS = 20
L2_NORMALIZATION = True

TOTAL_SAMPLES_TO_STREAM = 100_000_000  
NEW_SIGNAL_INTERVAL = 2_500_000       # Inject a new material every 2M pixels
n_batches = TOTAL_SAMPLES_TO_STREAM // PIXELS_PER_BATCH

# 2. MODEL INITIALIZATION

MFA_OTFP_model = MFA_OTFP(
    n_channels=num_bands,
    device="cpu", # Using CPU for fairer memory profiling (GPU memory is tracked differently)
    outlier_update_treshold=1000,
    q_max=5,
    L2_normalization=L2_NORMALIZATION
)

print("Training initial model...")
train_data = img_full[:N_TRAIN_ROWS, :, :].reshape(-1, num_bands)
MFA_OTFP_model.fit(torch.from_numpy(train_data).float())
print(f"Initial training complete. Components spawned: {MFA_OTFP_model.MFA.K}")

# 3. PROFILING SETUP
process = psutil.Process(os.getpid())

# Metrics to log
log_samples_seen = []
log_memory_mb = []
log_latency_ms = []
spawn_events = [] # Stores tuples of (samples_seen, current_K)

current_K = MFA_OTFP_model.MFA.K
samples_seen = 0
next_signal_threshold = NEW_SIGNAL_INTERVAL
anomaly_counter = 0

# 4. STREAMING LOOP
print(f"Starting stream for {n_batches} batches ({TOTAL_SAMPLES_TO_STREAM} pixels)...")

for batch_idx in tqdm(range(n_batches), desc="Processing Stream"):
    # 1. Fetch data
    start_row = (N_TRAIN_ROWS + batch_idx * BATCH_ROWS) % h
    
    # Handle wrap-around gracefully if we hit the bottom of the image
    if start_row + BATCH_ROWS > h:
        start_row = 0 
        
    batch_data = img_full[start_row:start_row+BATCH_ROWS, :, :].copy().reshape(-1, num_bands)
    
    # 2. Inject Novel Signal
    INJECT_SIGNAL = True
    if samples_seen >= next_signal_threshold and INJECT_SIGNAL:
        # 1. Generate a mathematically unique shape using increasing sine wave frequencies
        # Different frequencies ensure the shapes remain orthogonal after L2 normalization.
        x = np.linspace(0, 10, num_bands)
        freq = 1.0 + (anomaly_counter * 0.7) 
        base_signature = np.sin(x * freq) * 50 + 100 
        
        # 2. Create a dense cluster of 1500 pixels around this signature
        # We inject a small variance (sigma=2.0) so the MFA has a 3D volume to fit 
        # its latent factors (Lambda) and noise (Psi) to. 
        intra_class_variance = np.random.normal(0, 2.0, (1500, num_bands))
        novel_cluster = base_signature + intra_class_variance
        
        # 3. Inject it into the batch
        batch_data[0:1500, :] = novel_cluster
        
        next_signal_threshold += NEW_SIGNAL_INTERVAL
        anomaly_counter += 1
    # 3. Process Batch & Measure Latency
    batch_tensor = torch.from_numpy(batch_data).float()
    
    t0 = time.perf_counter()
    MFA_OTFP_model.process_data_block(batch_tensor) # Assuming this is your streaming function
    t1 = time.perf_counter()
    
    # 4. Measure Memory
    mem_mb = process.memory_info().rss / (1024 * 1024)
    
    # 5. Check for Component Spawns
    new_K = MFA_OTFP_model.MFA.K
    if new_K > current_K:
        spawn_events.append((samples_seen, new_K))
        current_K = new_K
        
    # 6. Log metrics
    samples_seen += PIXELS_PER_BATCH
    log_samples_seen.append(samples_seen)
    log_memory_mb.append(mem_mb)
    log_latency_ms.append((t1 - t0) * 1000) # Convert to milliseconds

# 5. PLOTTING
print("Stream finished. Generating plots...")

plt.style.use('seaborn-v0_8-whitegrid') # Clean academic style

# FIX 1: Make the figure wider (14x9 instead of 10x8)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# --- Plot 1: Memory Footprint ---
ax1.plot(log_samples_seen, log_memory_mb, color='tab:blue', alpha=0.8, linewidth=1.5)
ax1.set_ylabel("Memory Usage (MB)", fontsize=12, fontweight='bold')
ax1.set_title("OAMFA Hardware Viability: Memory & Latency over Infinite Stream", fontsize=14, fontweight='bold')

# Draw spawn lines and stagger text
for i, (spawn_x, k_val) in enumerate(spawn_events):
    ax1.axvline(x=spawn_x, color='red', linestyle='--', alpha=0.6)
    
    # FIX 2: Alternate the vertical position of the text so it doesn't overlap
    if i % 2 == 0:
        y_pos = max(log_memory_mb) * 0.95
    else:
        y_pos = max(log_memory_mb) * 0.85
        
    ax1.text(spawn_x, y_pos, f'K={k_val}', color='red', fontsize=10, rotation=90, va='top', ha='right')

# --- Plot 2: Processing Latency ---
ax2.plot(log_samples_seen, log_latency_ms, color='tab:orange', alpha=0.6, linewidth=1.0)
ax2.set_ylabel("Latency per Batch (ms)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Pixels Processed", fontsize=12, fontweight='bold')

# Draw spawn lines on the second plot too
for (spawn_x, k_val) in spawn_events:
    ax2.axvline(x=spawn_x, color='red', linestyle='--', alpha=0.6)

# Format X-axis to show millions (M)
formatter = FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}M')
ax2.xaxis.set_major_formatter(formatter)

# FIX 3: Manually define the padding instead of using tight_layout()
fig.subplots_adjust(hspace=0.2, top=0.92, bottom=0.1, left=0.08, right=0.95)

plt.savefig("hardware_viability_benchmark.png", dpi=300) # Save high-res for thesis
plt.show()

print(f"Final number of components: {current_K}")