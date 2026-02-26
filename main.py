import threading
import queue
import glob
import time
import torch
import os
from data_fetcher import producer, fetch_init_data, DataProduct
from otfp import MFA_OTFP


PERFORM_TIMING = True

## MODEL PARAMETERS
NUM_CHANNELS = 120                          
OUTLIER_SIGNIFICANCE = 0.01                 
OUTLIER_UPDATE_TRESHOLD = 3000               
Q_MAX = 10 

# Data parameters 
DATA_PRODUCT = DataProduct.L1B
N_SAMPLES_FIRST_MODEL = 15000 # Increased for better BIC initialization
IMAGE_PATHS = glob.glob(f'data/training_{DATA_PRODUCT.value}/*.nc')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

initial_batch = fetch_init_data(IMAGE_PATHS, N_SAMPLES_FIRST_MODEL, DATA_PRODUCT)

# Initialize the MFA-based OTFP model
MFA_OTFP_model = MFA_OTFP(
    init_data=initial_batch,
    n_channels=NUM_CHANNELS, 
    outlier_significance=OUTLIER_SIGNIFICANCE, 
    device=device, 
    outlier_update_treshold=OUTLIER_UPDATE_TRESHOLD,
    q_max=Q_MAX,
    L2_normalization=True
)

if __name__ == "__main__":
    
    start_time = 0
    if PERFORM_TIMING:
        start_time = time.perf_counter()
    
    seed = 42
    torch.manual_seed(seed)
    
    # Create a thread-safe queue, could calcuate maxsize based on data-info, for now just set it arbitrarirly

    queue = queue.Queue(maxsize=200) 

    # Start the producer thread
    producer_thread = threading.Thread(
        target=producer, 
        args=(IMAGE_PATHS, queue, DATA_PRODUCT, 5000),
        daemon=True
    )
    producer_thread.start()

    try:
        # Main processing loop
        n_processed_blocks = 0
        while True:
            
            projection_data = queue.get() 

            if projection_data == "FINISHED":
                print("\nSimulation finished.")
                break 
                
            projection_data = projection_data.to(device, non_blocking=True)
            
            # Stream the data block into your MFA model
            MFA_OTFP_model.process_data_block(X=projection_data)
            n_processed_blocks += 1
            if n_processed_blocks % 1000 == 0: 
                print(f"Processed; {n_processed_blocks} blocks of data")
            
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
            
    finally:
        # This will run whether the loop finishes or is interrupted
        print("\n--- Final Model Statistics ---")
        print(f"Total samples seen by model: {MFA_OTFP_model.n_samples_seen}")
        print(f"Total number of model updates: {MFA_OTFP_model.n_model_updates}")
        # You can add MFA-specific stats here later, like final K and q!
        
        if PERFORM_TIMING:
            print(f"Total processing time: {time.perf_counter() - start_time:.2f} seconds")
            print("Timing: ")
                
        # 1. Extract dynamic parameters directly from the trained model
        final_K = MFA_OTFP_model.MFA.K
        final_q = MFA_OTFP_model.MFA.q

        # 2. Build the state dictionary
        mfa_state = {
            'model_state_dict': MFA_OTFP_model.MFA.state_dict(), 
            
            # Hyperparameters required to initialize the MFA class
            'hyperparameters': {
                'n_components': final_K,
                'n_features': NUM_CHANNELS,
                'n_factors': final_q
            },
            
            # Streaming state required to resume the OTFP class later
            'streaming_state': {
                'll_threshold': getattr(MFA_OTFP_model, 'll_threshold', None),
                'n_samples_seen': MFA_OTFP_model.n_samples_seen
            }
        }

        # 3. Create a safe directory and dynamic filename
        save_dir = 'testing/models/'
        os.makedirs(save_dir, exist_ok=True) # Prevents crashes if folder doesn't exist
        
        save_path = f'{save_dir}/otfp_mfa.pt'
        
        # 4. Save to disk
        torch.save(mfa_state, save_path)
        print(f"\n MFA model successfully saved to '{save_path}'")