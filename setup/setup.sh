# install dependencies
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    screen \
    vim \
    iputils-ping


# Git clone this repo
git clone https://github.com/suzanne-stathatos/SAVeD.git
cd SAVeD

# Create a python venv
python -m venv saved
source saved/bin/activate   # On Linux/Mac

git submodule add https://github.com/JonathonLuiten/TrackEval.git

# Install python dependencies
pip install numpy opencv-python Pillow scikit-image wandb
pip install -r requirements.txt

# Log in to wandb
wandb login

# Start screen session for experiment
screen -S <experiment_name>

source saved/bin/activate

# Follow the steps in setup_{dataset}.sh to install the data for CFC22, CVA, POCUS, and Fluo
# Follow remaining setup steps in denoise_runs.sh and detect_runs.sh
