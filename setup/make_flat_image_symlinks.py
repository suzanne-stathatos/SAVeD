# Symlink images and flatten them
import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # make sure the input directory exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist")
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    # count the number of files recursively in the input directory
    input_num_files = 0
    for root, dirs, files in os.walk(input_dir, followlinks=True):
        input_num_files += len(files)
    print(f"Number of files in input directory: {input_num_files}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make symlinks for flat images
    for clip in os.listdir(input_dir):
        for file in os.listdir(os.path.join(input_dir, clip)):
            if file.endswith(".jpg") or file.endswith(".png"):
                # make the symlink
                path_to_symlink = os.path.join(input_dir, clip, file)
                path_to_symlink_target = os.path.join(output_dir, f'{clip}_{file}')
                os.symlink(path_to_symlink, path_to_symlink_target)

    # count the number of files in the output directory
    output_num_files = 0
    for root, dirs, files in os.walk(output_dir, followlinks=True):
        output_num_files += len(files)
    assert output_num_files == input_num_files, f"Number of files in output directory {output_dir}, {output_num_files}, does not match number of files in input directory {input_dir}, {input_num_files}"

if __name__ == "__main__":
    main()
