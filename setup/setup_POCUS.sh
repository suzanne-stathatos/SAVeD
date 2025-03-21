## Pocusvidnet 
git clone https://github.com/jannisborn/covid19_ultrasound.git
cd covid19_ultrasound

python -m venv pocusvidnet_env
source pocusvidnet_env/bin/activate   # On Linux/Mac
# Follow the instructions in their README to get baseline results

# Convert the videos to frames
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/convex --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/convex_baseline
python setup/convert_videos_to_frames.py --video-dir /root/covid19_ultrasound/data/pocus_videos/linear --output-dir /root/covid19_ultrasound/data/pocus_videos_frames/linear_baseline
# Make the frames median-filtered
python setup/classical_filter_frames_POCUS.py --frames-dir /root/covid19_ultrasound/data/pocus_videos_frames --output-dir /root/covid19_ultrasound/data/pocus_videos_frames_median --filter-type median

# Flatten the frames 
python setup/create_dataset_with_all_frames.py --frames_dir /root/covid19_ultrasound/data/pocus_videos_frames_median --frames_output_dir /root/covid19_ultrasound/data/pocus_videos_frames_median_flat
