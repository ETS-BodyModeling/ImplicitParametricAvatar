import os
import cv2
import numpy as np

def main():
    video_dir = "/home/fares/ImplicitParametricAvatar/output/output_animation_new"
    video_dir1= "/home/fares/ImplicitParametricAvatar/output_xavatar_simple/output_animation_new"
    output_dir = "/home/fares/ImplicitParametricAvatar/output"
    output_video_path = os.path.join(output_dir, 'animation1.mp4')

    # Ensure output directory exists
    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return
    
    video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
    video_files1 = sorted([os.path.join(video_dir1, f) for f in os.listdir(video_dir1) if os.path.isdir(os.path.join(video_dir1, f))])
    video_files = video_files + video_files1
    if not video_files:
        print("Error: No subdirectories found in the video directory.")
        return

    frame_rate = 120
    grid_rows, grid_cols = 4, 10
    frame_size = (512, 512)
    output_frame_size = (grid_cols * frame_size[0], grid_rows * frame_size[1])  # (3072, 2048)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, output_frame_size)

    for i in range(1996):
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
