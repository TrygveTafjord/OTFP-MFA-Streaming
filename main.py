import glob
import time
import torch
import os
from multiprocessing import Queue, Process
from data_fetcher import producer, fetch_init_data, DataProduct
from otfp import MFA_OTFP

PERFORM_TIMING = True

## MODEL PARAMETERS
NUM_CHANNELS = 120                          
OUTLIER_UPDATE_TRESHOLD = 2000               
Q_MAX = 8 

# Data parameters 
DATA_PRODUCT = DataProduct.L1B
N_SAMPLES_FIRST_MODEL = 10000 
IMAGE_PATHS = glob.glob(f'data/training_{DATA_PRODUCT.value}/*.nc')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    start_time = 0
    if PERFORM_TIMING:
        start_time = time.perf_counter()

    initial_batch = fetch_init_data(IMAGE_PATHS, N_SAMPLES_FIRST_MODEL, DATA_PRODUCT)

    # Initialize the MFA-based OTFP model
    MFA_OTFP_model = MFA_OTFP(
        init_data=initial_batch,
        n_channels=NUM_CHANNELS, 
        device=device, 
        outlier_update_treshold=OUTLIER_UPDATE_TRESHOLD,
        q_max=Q_MAX,
        L2_normalization=True
    )

    del initial_batch
    
    seed = 42
    torch.manual_seed(seed)
    
    # Create a thread-safe queue, could calcuate maxsize based on data-info, for now just set it arbitrarirly
    queue = Queue(maxsize=200) 

    # Start the producer thread
    camera_Hz = 22 #We assume we can batch 22 lines of data at a time, which corresponds to 1 second of flight. This can be adjusted based on actual data and performance needs.

    producer_thread = Process(
        target=producer, 
        args=(IMAGE_PATHS, queue, DATA_PRODUCT, camera_Hz),
        daemon=True
    )
    producer_thread.start()

    try:
        # Main processing loop
        n_processed_blocks = 0
        while True:
            
            projection_data = queue.get() 

            if isinstance(projection_data, str) and projection_data == "FINISHED":
                print("\nSimulation finished.")
                break

            projection_data = torch.tensor(projection_data, dtype=torch.float32).to(device, non_blocking=True)     
            projection_data = projection_data.to(device, non_blocking=True)
            
            # Stream the data block into your MFA model
            MFA_OTFP_model.process_data_block(X_3d=projection_data)
            n_processed_blocks += 1
            if n_processed_blocks % 1000 == 0: 
                print(f"==== Processed; {n_processed_blocks} blocks of data ====")
            
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
            
    finally:
        # This will run whether the loop finishes or is interrupted
        print("\n--- Final Model Statistics ---")
        print("="*55)
        print(" MFA SYSTEM STATE")
        print("="*55)

        # Model Architecture
        print(f"\n[MFA ARCHITECTURE]")
        print(f"Channels (D)                : {MFA_OTFP_model.MFA.D}")
        print(f"Latent Factors (q)          : {MFA_OTFP_model.MFA.q}")
        print(f"Total Components (K)        : {MFA_OTFP_model.MFA.K}")

        print(f"\n[COMPONENT BREAKDOWN]")
        print("-" * 55)

        # Convert tensors to CPU numpy arrays for easy string formatting
        pi_weights = torch.exp(MFA_OTFP_model.MFA.log_pi).detach().cpu().numpy()
        s0_mass = MFA_OTFP_model.MFA.S0.detach().cpu().numpy()
        updates = MFA_OTFP_model.MFA.update_counts.detach().cpu().numpy()
    
        for k in range(int(MFA_OTFP_model.MFA.K)):
            print(f"  ▶ Component {k}:")
            print(f"    ├─ Mixing Weight (π)       : {pi_weights[k]:.2%}")
            print(f"    ├─ Effective Pixel Mass    : {s0_mass[k]:,.1f} (from S0)")
            print(f"    └─ Stepwise EM Updates     : {int(updates[k])}\n")

        print(f"Total samples seen by model: {MFA_OTFP_model.n_samples_seen}")
        print(f"Total number of model cmponents: {MFA_OTFP_model.MFA.K}")
            # You can add MFA-specific stats here later, like final K and q!

        if PERFORM_TIMING:
            print(f"Total processing time: {time.perf_counter() - start_time:.2f} seconds")
                
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