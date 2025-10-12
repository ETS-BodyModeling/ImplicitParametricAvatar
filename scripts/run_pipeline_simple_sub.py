import os
import subprocess
import argparse
import time

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="run PifuHD")
    parser.add_argument('--output_path', type=str, default='output_sub/recons', help='Output directory for reconstructed mesh')
    parser.add_argument('--input_path', type=str, default='data/recon', help='Directory for target data')
    parser.add_argument('--output_path_render', type=str, default='output_sub/output_render', help='Output directory for rendered images')
    parser.add_argument('--run_animation', action='store_true', help='Run animation step')
    return parser.parse_args(args)

if __name__ == "__main__":
    # create output directory only if it doesn't exist
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # # run reconstruction code
    print("********running reconstruction  ----------- check output_sub/recons  *******")

    command = [
        "python", "-m", "scripts.main_optim_simple_sub",
        "--output_path", args.output_path, "--input_path", args.input_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Error:", result.stderr)

    # run rendering code
    print("********running rendering  ----------- check output_sub/output_render  *******")

    render_command = [
        "python", "-m", "scripts.main_render_simple_sub",
        "--output_path", args.output_path, "--output_path_render", args.output_path_render  
    ]
    render_result = subprocess.run(render_command, capture_output=True, text=True, check=False)
    print("Return code:", render_result.returncode)
    print("Output:", render_result.stdout)
    print("Error:", render_result.stderr)



    # run animation code if flag is set
    if args.run_animation:
        print("********running animation  ----------- check output_sub/animations  *******")
        animation_command = [
            "python", "-m", "scripts.main_animation_simple_sub",
            "--output_path", args.output_path, "--output_path_render", "output_sub/output_animation"
        ]
        animation_result = subprocess.run(animation_command, capture_output=True, text=True, check=False)
        print("Return code:", animation_result.returncode)
        print("Output:", animation_result.stdout)
        print("Error:", animation_result.stderr)


        # run combine video code
        print("********running combine video  ----------- check output_sub/output_combined  *******")

        combine_video_command = [
            "python", "-m", "scripts.combine_video", "--video_dir", "output_sub/output_animation", "--output_dir", "output_sub/output_combined"
        ]
        combine_video_result = subprocess.run(combine_video_command, capture_output=True, text=True, check=False)
        print("Return code:", combine_video_result.returncode)
        print("Output:", combine_video_result.stdout)
        print("Error:", combine_video_result.stderr)