import os
import subprocess
import json
import numpy as np

# Define the base command
base_command = "python eval.py --checkpoint data/epoch\=3200-test_mean_score\=1.000.ckpt --output_dir {output_dir} --device cuda:0 --guide_scaling_factor {scaling_factor}"

# Base output directory
base_output_dir = "data/realexp"
exp_output_dir = base_output_dir + "/realexp"

# Get input parameters
start_scaling_factor = float(input("Enter start scaling factor: "))
end_scaling_factor = float(input("Enter end scaling factor: "))
num_intervals = int(input("Enter number of intervals: "))

# Generate scaling factors
scaling_factors = np.linspace(start_scaling_factor, end_scaling_factor, num_intervals)

# Store results
results = []

# Iterate over different scaling factors
for scaling_factor in scaling_factors:
    output_dir = f"{exp_output_dir}_scale_{scaling_factor:.6f}"  # Name output dir
    
    # Ensure the output directory does not exist or remove it if needed
    if os.path.exists(output_dir):
        print(f"Warning: Output directory {output_dir} already exists. Overwriting...")
    
    # Format the command
    command = base_command.format(output_dir=output_dir, scaling_factor=scaling_factor)
    
    # Run the evaluation
    print(f"Running evaluation with guide_scaling_factor={scaling_factor:.6f}")
    subprocess.run(command, shell=True, check=True)
    
    # Load results from the generated JSON file
    json_path = os.path.join(output_dir, "eval_log.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            log_data = json.load(f)
            mean_score = log_data.get("test/mean_score", 0.0)  # Default to 0 if missing
            results.append((scaling_factor, mean_score))

# Sort results by mean score in descending order
results.sort(key=lambda x: x[1], reverse=True)

# Store ranked results in a text file
output_txt_path = os.path.join(base_output_dir, "ranked_scaling_factors.txt")
with open(output_txt_path, "w") as f:
    f.write("Ranked Scaling Factors by Mean Score:\n")
    for rank, (scaling_factor, score) in enumerate(results, start=1):
        f.write(f"{rank}. Scaling Factor: {scaling_factor:.6f}, Mean Score: {score:.6f}\n")

# Print the ranked list
print("\nRanked Scaling Factors by Mean Score:")
for rank, (scaling_factor, score) in enumerate(results, start=1):
    print(f"{rank}. Scaling Factor: {scaling_factor:.6f}, Mean Score: {score:.6f}")
