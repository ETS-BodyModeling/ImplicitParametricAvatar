import os
import subprocess
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="run PifuHD")
    parser.add_argument('--output_path', type=str, default='output', help='Output directory for reconstructed mesh')
    parser.add_argument('--input_path', type=str, default='data/demo', help='Directory for target data')
    return parser.parse_args(args)

if __name__ == "__main__":

    # run pifuhd
    args = parse_args()
    output_path_pifu = os.path.join(args.output_path,     command = [
        "python", "-m", "scripts.run_pifuhd",
        "--input_path", args.input_path,
        "--output_path", output_path_pifu,

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)

    # run reconstruction code

    command = [
        "python", "-m", "scripts.main_optim_1",

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)"output-pifu")


    # run animation code
    command = [
        "python", "-m", "scripts.main_animation",

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)