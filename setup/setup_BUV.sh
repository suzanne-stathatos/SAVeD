## CVA-Net
git clone https://github.com/jhl-Det/CVA-Net.git
cd CVA-Net
# on a virtual machine, follow their README instructions
# initialize conda
eval "$(conda shell.bash hook)"
conda create -n cva_net python=3.7 pip
conda activate cva_net
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# add rust to PATH
source $HOME/.cargo/env
# install separately 
pip install --only-binary :all: safetensors
pip install -r requirements.txt


# then before compiling the CUDA operators, install the CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo apt update
sudo apt install g++-9 gcc-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo sh cuda_11.3.0_465.19.01_linux.run --toolkit --override
sudo apt update
sudo apt install build-essential
sudo apt install g++-9  # or another compatible version
export CXX=/usr/bin/g++-9
export CC=/usr/bin/gcc-9

# Add CUDA to PATH and LD_LIBRARY_PATH in ~/.bashrc

# go through the test of setup
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py

# Download CVA-Net data
pip install gdown
gdown https://drive.google.com/uc?id=1LVXK34OJhC2LkqqyMmVFnXsXQavvZdeF
mkdir -p "/root/CVA-Net/miccai_buv_baseline"
# un7z the data
sudo apt install p7zip-full p7zip-rar
7z x "Miccai 2022 BUV Dataset.7z" -o"/root/CVA-Net/miccai_buv_baseline"

# Also download the pretrained model
gdown https://drive.google.com/uc?id=1Wqlh0gBgbzWrXZEcjrPhRqRZ1lmf33iP
