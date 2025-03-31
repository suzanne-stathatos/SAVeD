# From Claude-3.7-sonnet-20250324

#!/usr/bin/env python3
import sys
from PIL import Image
import argparse

def create_gif(image_paths, output_path, duration=1000):
    """
    Create a GIF from a list of images.
    
    Args:
        image_paths (list): List of paths to the input images
        output_path (str): Path where the output GIF will be saved
        duration (int): Duration of each frame in milliseconds (default: 500ms)
    """
    images = []
    
    # Open each image and append to list
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            sys.exit(1)
    
    # Check if we have images to process
    if not images:
        print("No valid images provided")
        sys.exit(1)
    
    # Save as GIF
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=duration,
            loop=0  # 0 means loop forever
        )
        print(f"GIF created successfully: {output_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from 3 images")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("image3", help="Path to third image")
    parser.add_argument("-o", "--output", default="output.gif", help="Output GIF filename (default: output.gif)")
    parser.add_argument("-d", "--duration", type=int, default=500, help="Duration of each frame in milliseconds (default: 500)")
    
    args = parser.parse_args()
    
    create_gif([args.image1, args.image2, args.image3], args.output, args.duration)