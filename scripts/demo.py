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
    print("********running pifuhd ----------- check output/output-pifu *******")
    args = parse_args()
    output_path_pifu = os.path.join(args.output_path, "output-pifu")
    command = [
        "python", "-m", "scripts.run_pifuhd",
        "--input_path", args.input_path,
        "--output_path", output_path_pifu,

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)

    # run reconstruction code
    print("********running reconstruction  ----------- check output/recons  *******")

    command = [
        "python", "-m", "scripts.main_optim",

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)

    # run animation code
    print("********running animation  ----------- check output/output-animation *******")
    command = [
        "python", "-m", "scripts.main_animation",

    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)