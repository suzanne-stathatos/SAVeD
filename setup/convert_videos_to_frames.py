import cv2
import os
import argparse

def extract_frames(video_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all video files in the directory
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        
        # Get the base name of the video file (with extension)
        base_name = video_file
        
        # Create a directory for the frames of this video
        frame_dir = os.path.join(output_dir, base_name)
        os.makedirs(frame_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save the frame as a PNG file
            frame_filename = os.path.join(frame_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} to {frame_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert videos to frames.")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save frames")
    args = parser.parse_args()

    extract_frames(args.video_dir, args.output_dir)

if __name__ == "__main__":
    main()