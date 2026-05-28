from hypso.hypso1 import Hypso1
from hypso.write import write_l1d_nc_file, write_l1b_nc_file
import glob

data_dir = []
data_dir.append(glob.glob('data/new_data/*.nc'))
#data_dir.append(glob.glob('data/linearity_test/*.nc'))
print(f"Found {len(data_dir)} files.")

for sublist in data_dir:
    # Process each L1A file
    for file_name in sublist:
        try:
            satobj = Hypso1(path=file_name, verbose=True)
            satobj.generate_l1b_cube()
            #satobj.generate_l1c_cube()
            #satobj.generate_l1d_cube()
    
            #write_l1d_nc_file(satobj=satobj, overwrite=False)
            #print(f":white_tick: Successfully created L1D file for: {file_name}")
            write_l1b_nc_file(satobj=satobj, overwrite=True)
            print(f":white_tick: Successfully created L1B file for: {file_name}")

        except Exception as e:
            print(f":x: Error processing {file_name}: {e}")