# PointMax in OpenShape
We use PointNeXt for zero sample classification based on OpenShape and embed our PointMax

# Preparation 
Please refer to [here](https://github.com/Colin97/OpenShape_code.git) to prepare the data and environment

OpenShape uses `src/configs/train.yml` to control the model to be used. Take PointNeXt as an example, set the PointNeXt version to be used by scaling, where 1 is pointnext-s, 2 is pointnext-b, and 3 is pointnext-L. 4 indicates PointNeXt_PointMax. For details, see `src\models\pointnext.py`

# Model Zoo
|      Model      | Training Data |                                Objaverse-LVIS Zero-Shot Top1 (Top5)                                | FLOPs (G) |
| :-------------: | :-----------: | :------------------------------------------------------------------------------------------------: | :-------: |
|   PointNeXt-S   | ShapeNet Only | [10.5(25.2)](https://drive.google.com/drive/folders/1QqGqZMbLUvbEAqd_j75rcajSe_-eU_a5?usp=sharing) |    4.27   |
| +PointMax (0.5) | ShapeNet Only | [10.8(25.2)](https://drive.google.com/drive/folders/1hqEJUYXYPZr-XTABI5j-thBEaMIrqBch?usp=sharing) |    3.36   |
| +PointMax (0.7) |      ShapeNet Only         |       [10.8(25.3)](https://drive.google.com/drive/folders/1oUN6Wbu6HpaF7AV_eK-q-Ty879QYYQCH?usp=sharing)                                                                                          |        2.79   |
# Train

``` 
python3 src/main.py
```

# Acknowledgment
This repository is built on reusing codes of [OpenShape](https://github.com/Colin97/OpenShape_code/tree/master)