# PointMax
This is a official implementation of PointMax proposed by our paper:  "PointMax: Self-Boosted Local Sampling for 3D Point Cloud Analysis"

# Tasks
We currently provide implementations on classification tasks:

 1.[PointNet++](./PointNet++) for point cloud classification on ScanObjectNN.
 2. [OpenShape](./OpenShape_code) for zero-shot classification on Objaverse-LVIS.

# Install 
``` 
cd PointMetaBase

source install.sh
```
If you want to use OpenShape, install the following package
```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine

conda install -c dglteam/label/cu113 dgl

pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm open3d 
```
# Dataset

 1. The processed training and evaluation data  of  OpenShape zero-shot classification can be found [here](https://huggingface.co/datasets/OpenShape/openshape-training-data)
 2.  ScanObjectNN can be downloaed in [here](https://guochengqian.github.io/PointNeXt/examples/scanobjectnn/)
   
   
 # Acknowledgment
 This repository is built on reusing codes of  [PointMetaBase](https://github.com/linhaojia13/PointMetaBase.git) [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)and  [OpenShape](https://github.com/Colin97/OpenShape_code.git)
 
 