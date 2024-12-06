import subprocess

# Define the paths
input_path = "input_path"
output_path = "output_path"
recon_path = f"{output_path}/pifuhd_final/recon"

# Command 1: Run the simple_test script
cmd1 = [
    "python", 
    "-m", 
    "submodules.pifuhd.apps.simple_test", 
    "-i", input_path, 
    "-o", output_path, 
    "-r", "512"
]

# Command 2: Run the clean_mesh script
cmd2 = [
    "python", 
    "apps/clean_mesh.py", 
    "-f", recon_path
]

# Execute the commands
try:
    print("Running simple_test...")
    subprocess.run(cmd1, check=True)
    print("simple_test completed successfully.")
    
    print("Running clean_mesh...")
    subprocess.run(cmd2, check=True)
    print("clean_mesh completed successfully.")
    
except subprocess.CalledProcessError as e:
    print(f"Error: Command failed with return code {e.returncode}")
    print(e.output)
