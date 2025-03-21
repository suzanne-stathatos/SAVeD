## Noise2Void

# CAREamics for N2V2 and Noise2Void
# https://careamics.github.io/0.1/installation/
# Create a python virtual env 
python -m venv n2v_env
source n2v_env/bin/activate   # On Linux/Mac

# install basics including pytorch, etc
pip install numpy opencv-python Pillow scikit-image wandb
pip install -r ~/SAVeD/requirements.txt

# Verify the GPU is available
python -c "import torch; print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])"

# install careamics (with examples notebooks and wandb)
pip install "careamics[examples, wandb]"

# Setup training images and validation images directory for completely flat images
# CFC22
python setup/make_flat_image_symlinks.py --input_dir /path/to/Data/CFC22/frames/raw_all --output_dir /path/to/Data/CFC22/frames/raw_all_flat_symlinks
# POCUS
python setup/make_flat_image_symlinks.py --input_dir /path/to/covid19_ultrasound/data/image_dataset_baseline --output_dir /path/to/covid19_ultrasound/data/image_dataset_baseline_flat_symlinks
# CVA
python setup/create_dataset_with_all_frames.py --frames_dir /path/to/CVA-Net/miccai_buv/rawframes_baseline --frames_output_dir /path/to/CVA-Net/miccai_buv/rawframes_baseline_all
python setup/make_flat_image_symlinks.py --input_dir /path/to/CVA-Net/miccai_buv/rawframes_baseline_all --output_dir /path/to/CVA-Net/miccai_buv/rawframes_baseline_all_flat_symlinks

# To run n2v 
python ~/SAVeD/other_baselines/n2v_cfc22.py
python ~/SAVeD/other_baselines/n2v_POCUS.py
python ~/SAVeD/other_baselines/n2v_CVA.py
# To run n2v prediction
python ~/SAVeD/other_baselines/n2v_predict_cfc22.py
python ~/SAVeD/other_baselines/n2v_predict_POCUS.py
python ~/SAVeD/other_baselines/n2v_predict_CVA.py
deactivate
