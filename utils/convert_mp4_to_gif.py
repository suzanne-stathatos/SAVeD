import os
import cv2
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_mp4_to_gif(mp4_file, gif_file, remove_mp4=False, fps=20):
    """
    Convert MP4 file to GIF file.
    
    :param mp4_file: Path to the MP4 file
    :param gif_file: Path to the GIF file
    :param remove_mp4: Whether to remove the MP4 file after conversion
    :param fps: Frames per second for the output GIF (default: 20)
    """
    output_folder = os.path.dirname(gif_file)
    os.makedirs(output_folder, exist_ok=True)

    input_path = mp4_file
    output_path = gif_file

    # Open the video file
    video = cv2.VideoCapture(input_path)
    
    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read frames
    frames = []
    for _ in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize frame to be half of its original size
        frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
        frames.append(frame_rgb)
    
    # Release the video capture object
    video.release()

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)

    if remove_mp4:
        # print(f'Removing {input_path}')
        os.remove(input_path)

    print(f"Conversion complete. GIF saved in {gif_file}")

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('--mp4_file', type=str, required=True, help='path to mp4')
    # parser.add_argument('--gif_file', type=str, required=True, help='path to gif')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # convert_mp4_to_gif(args.mp4_file, args.gif_file)
    convert_mp4_to_gif('bsf2f_kenai-channel_denoised/bg_sub_2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732.mp4', 'bsf2f_kenai-channel_denoised/bg_sub_2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732.gif')
