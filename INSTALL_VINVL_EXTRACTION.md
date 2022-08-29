## Installation

### Requirements:
- PyTorch 1.7
- torchvision
- cocoapi
- yacs>=0.1.8
- numpy>=1.19.5
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 10.1


### 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name vinvl_fx python=3.7 -y
conda activate vinvl_fx

# this installs the right pip and dependencies for the fresh python
conda install ipython h5py nltk joblib jupyter pandas scipy

# maskrcnn_benchmark and coco api dependencies
pip install ninja "yacs>=0.1.8" cython matplotlib tqdm opencv-python "numpy>=1.19.5"

conda install cudatoolkit=11.0 -c pytorch
pip install pytorch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2+cu110
pip install timm einops
# install pycocotools
conda install -c conda-forge pycocotools

# install cityscapesScripts
python -m pip install cityscapesscripts

# install Scene Graph Detection
git clone https://github.com/microsoft/scene_graph_benchmark
cd scene_graph_benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
```

### 2: Download relevant files

```bash
mkdir pretrained_model
cd pretrained_model
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
cd ..
mkdir datasets && cd datasets
mkdir visualgenome && cd visualgenome
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json

```
