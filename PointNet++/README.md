# Pytorch Implementation of PointMax on PointNet++ 

We implemented PointMax on classification tasks. For details, please refer to `models\pointnet2_cls_ssg.py` and set `pointmax=True` to enable PointMax


# Run

``` 
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg

python test_classification.py --log_dir pointnet2_cls_ssg
```

# Acknowledgment
This repository is built on reusing codes of [PointNet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)