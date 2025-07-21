import subprocess
from pathlib import Path
import sys
import argparse
import os
# this is a stat file to evaluate the results of the 3d and 2d reconstruction
_ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_ROOT_DIR))
def main(input_path, output_path):
    # Define the paths
    # input_path = "input_path"
    # output_path = "output_path"
    # recon_path = f"{output_path}/pifuhd_final"

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
        "-m",
        "submodules.pifuhd.apps.clean_mesh",
        "-f", output_path
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

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="run PifuHD")
    parser.add_argument('--output_path', type=str, default='output_pifu', help='Output directory for reconstructed mesh')
    parser.add_argument('--input_path', type=str, default='data/openpose', help='Directory for target data')
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    root_path = _ROOT_DIR.parent
    input_path = os.path.join(root_path, args.input_path)
    output_path = os.path.join(root_path, args.output_path)
    # Check if path exists, create if it doesn't
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(input_path, output_path)