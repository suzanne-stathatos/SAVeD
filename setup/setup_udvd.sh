## UDVD

# Create a python virtual env 
python -m venv udvd_env
source udvd_env/bin/activate   # On Linux/Mac

# install basics including pytorch, etc
pip install numpy opencv-python Pillow scikit-image wandb h5py
pip install -r requirements.txt

# Download the UDVD model
git clone https://github.com/sreyas-mohan/udvd.git ../other_denoisers/udvd
cd ../other_denoisers/udvd

