import cv2
import os
import shutil
import glob
from pathlib import Path
__all__ = ['make_mp4_from_images']

def make_mp4_from_images(clip_path, output_path):
    '''
    Get video sequence length and h,w of images
    Returns:
        frames: a list of image paths
        img_size: size of the images in the video (h,w,c)
    '''
    frames = sorted(glob.glob(os.path.join(clip_path, '*.png')) + glob.glob(os.path.join(clip_path, '*.jpg')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # Create an MP4 video from the sequence of images    
    output_path = os.path.join(output_path, f'{Path(clip_path).name}.mp4')
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    
    # print(f"MP4 video created at: {output_path}")
    return output_path, frame.shape, len(frames)    


if __name__ == '__main__':
    DATA_ROOT = '/path/to/Data/CFC22'
    video_root = os.path.join(DATA_ROOT, 'frames_bgsubf2f_mp4s')
    if os.path.exists(video_root):
        shutil.rmtree ( video_root )
    
    clip_path = os.path.join(DATA_ROOT, 'frames', 'background_subtracted_frame_to_frame_difference', 'kenai-channel/2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732')
    output_path = os.path.join(video_root, 'kenai-channel')
    os.makedirs ( output_path, exist_ok=True )
    # make_mp4_from_images('clip_path', output_path)
    os.makedirs('bsf2f_kenai-channel_denoised', exist_ok=True)
    make_mp4_from_images('/path/to/data/kenai-channel/bg_sub_2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732', 'bsf2f_kenai-channel_denoised')
