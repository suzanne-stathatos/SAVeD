# This assumes that `setup.sh` has already been run

# Download the fish dataset
mkdir -p Data/CFC22
cd Data/CFC22
wget https://data.caltech.edu/records/1y23m-j8r69/files/fish_counting_frames.tar.gz
wget https://data.caltech.edu/records/1y23m-j8r69/files/fish_counting_annotations.tar.gz
wget https://data.caltech.edu/records/1y23m-j8r69/files/fish_counting_metadata.tar.gz
tar -xvf fish_counting_frames.tar.gz
tar -xvf fish_counting_annotations.tar.gz
tar -xvf fish_counting_metadata.tar.gz
find . -name .ipynb_checkpoints -exec rm -rf {} +
mv annotations/ annotations_mot/
cd ../..

# Create a dataset with all frames from all domains, using symlinks
python setup/create_dataset_with_all_frames.py --frames_dir Data/CFC22/frames/raw --annotations_dir Data/CFC22/annotations_mot --frames_output_dir Data/CFC22/frames/raw_all --annotations_output_dir Data/CFC22/annotations_mot_all

# Background subtracted frames
python setup/background_subtract_frames.py --frames-dir Data/CFC22/frames/raw --output-dir Data/CFC22/frames/raw_bs
python setup/create_dataset_with_all_frames.py --frames_dir Data/CFC22/frames/raw_bs --annotations_dir Data/CFC22/annotations_mot --frames_output_dir Data/CFC22/frames/raw_bs_all --annotations_output_dir Data/CFC22/annotations_mot_all

# Median filtered frames
python setup/classical_filter_frames.py --frames-dir /path/to/Data/CFC22/frames/raw --output-dir /path/to/Data/CFC22/frames/raw_median --filter-type median