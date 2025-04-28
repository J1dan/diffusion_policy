import os
import subprocess
import json
import numpy as np

# Get input parameters
start_scaling_factor = float(input("Enter start scaling factor: "))
end_scaling_factor = float(input("Enter end scaling factor: "))
num_intervals = int(input("Enter number of intervals: "))

# Generate scaling factors
scaling_factors = np.linspace(start_scaling_factor, end_scaling_factor, num_intervals)

# Define the parent output directory based on start/end scaling factors
base_output_dir = f"data/elevatewall/random2_test_{start_scaling_factor:.8f}-{end_scaling_factor:.8f}"
os.makedirs(base_output_dir, exist_ok=True)  # Create parent directory if it doesn't exist

# Define the base command with placeholders
base_command = (
    "python eval.py "
    "--checkpoint ../../diffusion_policy_a/data/outputs/test1800_smooth/checkpoints/epoch\=0050-test_mean_score\=0.090.ckpt "
    "--output_dir {output_dir} "
    "--device cuda:0 "
    "--guide_scaling_factor {scaling_factor}"
)

# base_command = (
#     "python eval.py "
#     "--checkpoint ../../diffusion_policy_a/data/outputs/test900_smooth/checkpoints/epoch\=0250-test_mean_score\=0.065.ckpt "
#     "--output_dir {output_dir} "
#     "--device cuda:0 "
#     "--guide_scaling_factor {scaling_factor}"
# )

# Store results
results = []

# Iterate over different scaling factors
for scaling_factor in scaling_factors:
    # Create a subfolder for each scaling factor
    output_dir = os.path.join(base_output_dir, f"scale_{scaling_factor:.8f}")
    # os.makedirs(output_dir, exist_ok=True)

    # Format the command
    command = base_command.format(output_dir=output_dir, scaling_factor=scaling_factor)

    # Run the evaluation
    print(f"Running evaluation with guide_scaling_factor={scaling_factor:.8f}")
    subprocess.run(command, shell=True, check=True)

    # Load results from the generated JSON file
    json_path = os.path.join(output_dir, "eval_log.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            log_data = json.load(f)
            mean_score = log_data.get("test/mean_score", 0.0)
            results.append((scaling_factor, mean_score))

# Sort results by mean score in descending order
results.sort(key=lambda x: x[1], reverse=True)

# Store ranked results in a text file under the parent folder
output_txt_path = os.path.join(base_output_dir, "ranked_scaling_factors.txt")
with open(output_txt_path, "w") as f:
    f.write("Ranked Scaling Factors by Mean Score:\n")
    for rank, (scaling_factor, score) in enumerate(results, start=1):
        f.write(f"{rank}. Scaling Factor: {scaling_factor:.8f}, Mean Score: {score:.8f}\n")

# Print the ranked list
print("\nRanked Scaling Factors by Mean Score:")
for rank, (scaling_factor, score) in enumerate(results, start=1):
    print(f"{rank}. Scaling Factor: {scaling_factor:.8f}, Mean Score: {score:.8f}")
