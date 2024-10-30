# PointMetaBase
We mainly provide the configuration of PointMax embedded into PointNeXt, please check
```
cfgs\scanobjectnn\pointnext_pointmax.yaml

cfgs\modelnet40ply2048\pointnext_pointmax.yaml
```

**The code for scene segmentation will be released soon!** 
# ScanObjectNN
TP: Throughput (instance per second) measured using an NVIDIA GeForce GTX 1080Ti GPU and two 8 core Intel(R) Xeon(R) @ 2.10GHz CPU.
|    Model    |                                                   OA/mAcc                                                   | FLOPs (G) |TP (ins./sec.)|
| :---------: | :---------------------------------------------------------------------------------------------------------: | :-------: | :-------: 
| PointNeXt-S | [87.7±0.4 / 85.8±0.6](https://drive.google.com/drive/folders/1A584C9x5uAqppbjNNiVqlA_7uOOOlEII?usp=sharing) |    1.64   |442|
|  +PointMax (0.5)  |  [87.4/85.73](https://drive.google.com/drive/folders/1U-gKF9olvn678YcuKrykzbcikGQJFEAK?usp=sharing)      |1.04|522|                                                                            

## Train

``` 
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext_pointmax.yaml wandb.use_wandb=False

```
## Test

``` 
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/pointnext_pointmax.yaml  mode=test --pretrained_path path/to/pretrained/model
```

## Profile Parameters, FLOPs, and Throughput

``` 
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/pointnext_pointmax.yaml batch_size=32 num_points=1024 timing=True flops=True
```

# Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints/tree/baeca5e319aa2e756d179e494469eb7f5ffd29f0) [,PointNeXt](https://github.com/guochengqian/PointNeXt.git) and [PointMeta](https://github.com/linhaojia13/PointMetaBase.git).

