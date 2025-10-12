
import os
import cv2
import numpy as np
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Combine videos from directories into a grid video.")
    parser.add_argument('--video_dir', type=str, required=False, default='/home/fares/ImplicitParametricAvatarRefined/output_simple/output_animation', help='First directory containing video subfolders')
    # parser.add_argument('--video_dir1', type=str, required=False, default='/path/to/default/video_dir1', help='Second directory containing video subfolders')
    parser.add_argument('--output_dir', type=str, required=False, default='/home/fares/ImplicitParametricAvatarRefined/output_simple/output_combined', help='Directory to save the output video')
    parser.add_argument('--output_video_name', type=str, default='animation_tot.mp4', help='Name of the output video file')
    parser.add_argument('--num_frames', type=int, default=1996, help='Number of frames to process')
    parser.add_argument('--frame_rate', type=int, default=120, help='Frame rate of the output video')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of grid rows')
    parser.add_argument('--grid_cols', type=int, default=5, help='Number of grid columns')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[512, 512], help='Size of each frame (width height)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    video_dir = args.video_dir
    # video_dir1 = args.video_dir1
    output_dir = args.output_dir
    output_video_path = os.path.join(output_dir, args.output_video_name)
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
    # video_files1 = sorted([os.path.join(video_dir1, f) for f in os.listdir(video_dir1) if os.path.isdir(os.path.join(video_dir1, f))])
    # video_files = video_files + video_files1
    if not video_files:
        print("Error: No subdirectories found in the video directories.")
        return

    frame_rate = args.frame_rate
    grid_rows, grid_cols = args.grid_rows, args.grid_cols
    frame_size = tuple(args.frame_size)
    output_frame_size = (grid_cols * frame_size[0], grid_rows * frame_size[1])

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, output_frame_size)

    for i in range(args.num_frames):
        print(f"Processing frame {i}...")
        imgs = []

        for path_vid in video_files:
            image_path = os.path.join(path_vid, f'{i:04}.png')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, frame_size)
                    imgs.append(resized_image)

        if not imgs:
            print(f"Warning: No images found for frame {i}. Skipping.")
            continue

        # Create blank image for combined frame
        combined_image = np.zeros((output_frame_size[1], output_frame_size[0], 3), dtype=np.uint8)

        # Fill the grid with images
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx < len(imgs):
                    combined_image[r*frame_size[1]:(r+1)*frame_size[1], c*frame_size[0]:(c+1)*frame_size[0]] = imgs[idx]
        print(combined_image.shape)

        output_video.write(combined_image)

    output_video.release()
    print(f"Video saved at {output_video_path}")


if __name__ == "__main__":
    main()
