#!/usr/bin/env python3

import os
import argparse
from PIL import Image
import glob
import numpy as np

def convert_tiff_to_jpg(input_dir, output_dir=None, quality=90):
    # If output directory is not specified, use input directory
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all TIFF files
    tiff_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    
    if not tiff_files:
        print(f"No TIFF files found in '{input_dir}'.")
        return
    
    print(f"Found {len(tiff_files)} TIFF files. Converting to JPEG with quality {quality}...")
    
    # Process each TIFF file
    for i, tiff_file in enumerate(tiff_files, 1):
        # Get the filename without extension
        basename = os.path.splitext(os.path.basename(tiff_file))[0]
        
        # Create output path
        output_path = os.path.join(output_dir, f"{basename}.jpg")
        
        try:
            # Open and convert the image
            with Image.open(tiff_file) as img:
                # Convert to RGB if necessary (TIFF might be CMYK or other modes)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # make the image range between 0 and 255
                img = np.array(img)
                # normalize the image to be between 0 and 1
                min_val = np.min(img)
                max_val = np.max(img)
                img = (img - min_val) / (max_val - min_val)
                img = img * 255
                img = Image.fromarray(img.astype(np.uint8))
                
                # Save as JPEG
                img.save(output_path, "JPEG", quality=quality)
            
            print(f"Progress: {i}/{len(tiff_files)} files converted", end="\r")
        
        except Exception as e:
            print(f"Error converting {tiff_file}: {e}")
    
    print(f"\nConversion complete! {len(tiff_files)} TIFF files converted to JPEG.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF files to JPEG format")
    parser.add_argument("--input_dir", help="Directory containing TIFF files")
    parser.add_argument("--output_dir", help="Output directory for JPEG files")
    parser.add_argument("--quality", type=int, default=100, help="JPEG quality (1-100, default: 90)")
    
    args = parser.parse_args()

    for subdir in os.listdir(args.input_dir):
        if os.path.isdir(os.path.join(args.input_dir, subdir)):
            convert_tiff_to_jpg(os.path.join(args.input_dir, subdir), os.path.join(args.output_dir, subdir), args.quality)
