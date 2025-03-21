# Fluo
sudo apt install unzip
mkdir -p /root/Fluo-C2DL-MSC
wget http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip
unzip -o Fluo-C2DL-MSC.zip -d /root/Fluo-C2DL-MSC/train
wget http://data.celltrackingchallenge.net/test-datasets/Fluo-C2DL-MSC.zip
unzip -o Fluo-C2DL-MSC.zip -d /root/Fluo-C2DL-MSC/test

mkdir -p /root/Fluo-N2DH-GOWT1
wget http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip
unzip -o Fluo-N2DH-GOWT1.zip -d /root/Fluo-N2DH-GOWT1/train
wget http://data.celltrackingchallenge.net/test-datasets/Fluo-N2DH-GOWT1.zip
unzip -o Fluo-N2DH-GOWT1.zip -d /root/Fluo-N2DH-GOWT1/test

mkdir -p /root/Fluo/train
mkdir -p /root/Fluo/test
cp -r /root/Fluo-C2DL-MSC/train/Fluo-C2DL-MSC/01 /root/Fluo/train/Fluo-C2DL-MSC_01
cp -r /root/Fluo-C2DL-MSC/train/Fluo-C2DL-MSC/02 /root/Fluo/train/Fluo-C2DL-MSC_02
cp -r /root/Fluo-N2DH-GOWT1/train/Fluo-N2DH-GOWT1/01 /root/Fluo/train/Fluo-N2DH-GOWT1_01
cp -r /root/Fluo-N2DH-GOWT1/train/Fluo-N2DH-GOWT1/02 /root/Fluo/train/Fluo-N2DH-GOWT1_02
cp -r /root/Fluo-C2DL-MSC/test/Fluo-C2DL-MSC/01 /root/Fluo/test/Fluo-C2DL-MSC_01
cp -r /root/Fluo-C2DL-MSC/test/Fluo-C2DL-MSC/02 /root/Fluo/test/Fluo-C2DL-MSC_02
cp -r /root/Fluo-N2DH-GOWT1/test/Fluo-N2DH-GOWT1/01 /root/Fluo/test/Fluo-N2DH-GOWT1_01
cp -r /root/Fluo-N2DH-GOWT1/test/Fluo-N2DH-GOWT1/02 /root/Fluo/test/Fluo-N2DH-GOWT1_02

python setup/classical_filter_frames_Fluo.py --frames-dir /root/Fluo --output-dir /root/Fluo_median --filter-type median
python setup/create_dataset_with_all_frames.py --frames_dir /root/Fluo_median --frames_output_dir /root/Fluo_median_flat
