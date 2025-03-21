# Essentially the opposite of convert_videos_to_frames.py
import os
import argparse
import cv2
from PIL import Image   


def convert_frames_to_videos(dir_of_frames_dir, video_dir):
    # Create output directory if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)

    # Loop over all frames directories in the frames_dir
    for frame_dir in os.listdir(dir_of_frames_dir):
        frame_dir_path = os.path.join(dir_of_frames_dir, frame_dir)
        # get the video name from the frame_dir
        video_name = frame_dir # expects the frame_dir to be the video name, including the extension
        # create the video path
        video_path = os.path.join(video_dir, f"{video_name}")
        # get the frames from the frame_dir
        frames = [os.path.join(frame_dir_path, f) for f in os.listdir(frame_dir_path) if f.endswith('.png')]
        # sort the frames
        frames.sort()
        # create the video
        video_suffix = video_name.split(".")[-1]
        codec = 'XVID'
        if video_suffix == 'mp4':
            codec = 'mp4v'
        elif video_suffix == 'MP4':
            codec = 'mp4v'
        elif video_suffix == 'avi':
            codec = 'XVID'
        elif video_suffix == 'mpeg':
            codec = 'XVID'
        elif video_suffix == 'mov':
            codec = 'mp4v'
        elif video_suffix == 'gif':
            codec = None
        else:
            raise ValueError(f"Video suffix {video_suffix} not supported")

        if video_suffix != 'gif':
            print(video_suffix)
            print(video_name)
            frame_shape = cv2.imread(frames[0]).shape
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*codec), 10, (frame_shape[1], frame_shape[0]))
            for frame in frames:
                video_writer.write(cv2.imread(frame))
            video_writer.release()
        else:
            # create a gif
            gif_frames = []
            for frame in frames:
                gif_frames.append(Image.open(frame))
            gif_frames[0].save(video_path, save_all=True, append_images=gif_frames[1:])
        print(f"Created video {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert frames to videos.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frames")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory to save videos")
    args = parser.parse_args()

    convert_frames_to_videos(args.frames_dir, args.video_dir)

if __name__ == "__main__":
    main()