import h5py
import re
from termcolor import colored

dataset_path = "data/robomimic/datasets/can/custom/lowdim123.hdf5"

with h5py.File(dataset_path, "a") as f:
    # Extract and sort demo keys correctly
    demos = sorted(
        [key for key in f['data'].keys() if key.startswith("demo")],
        key=lambda x: int(re.search(r'\d+', x).group())  # Extract numeric part safely
    )

    # Rename each demo to "demo_x" where x starts from 0
    for i, old_name in enumerate(demos):
        new_name = f"demo_{i}"  # Format: demo_x
        f['data'].move(old_name, new_name)
        print(colored(f"Renamed '{old_name}' -> '{new_name}'", "yellow"))
