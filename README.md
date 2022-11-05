# The Implementation with MindSpore of CVPR 2022 MAVOC Challenge's Champion Proposal

**Paper Link:**   https://arxiv.org/abs/2205.01920

**Award certificate**

<p align="left">
  <img src="./tutorials/MAVOC_Track 1_Champion.png" width=360 />
</p>

<p align="left">
  <img src="./tutorials/MAVOC_Track 2_Champion.png" width=360 />
</p>

## Introduction
Multi-modal aerial view object classification (MAVOC) in Automatic target recognition (ATR), although an important and challenging problem, has been under studied. This paper firstly finds that fine-grained data, class imbalance and various shooting conditions preclude the representational ability of general image classification. Moreover, the MAVOC dataset has scene aggregation characteristics. By exploiting these properties, we propose Scene Clustering Based Pseudo-labeling Strategy (SCP-Label), a simple yet effective method to employ in post-processing. The SCP-Label brings greater accuracy by assigning the same label to objects within the same scene while also mitigating bias and confusion with model ensembles. Its performance surpasses the official baseline by a large margin of +20.57% Accuracy on Track 1 (SAR), and +31.86% Accuracy on Track 2 (SAR+EO), demonstrating the potential of SCP-Label as post-processing. Finally, we win the championship both on Track1 and Track2 in the CVPR 2022 Perception Beyond the Visible Spectrum (PBVS) Workshop MAVOC Challenge.
	
### Experiments Results

| Backbone            | Under-sample| Augmentation Method       | Top-1 Accuracy (%) |
|:--------------------|:------------|:------------              |:------------       |
| Resnet50            | all data    | Rotation+Flipping+Cutmix  | 17.10              |
| Efficientnet-b1     | all data    | Rotation+Flipping+Cutmix  | 15.88              |
| Swin-Transformer    | all data    | Rotation+Flipping+Cutmix  | 16.88              |
| DenseNet161         | all data    | Rotation+Flipping+Cutmix  | 18.05              |
| MobileNetV3-large   | 1741        | Rotation+Flipping+Cutmix  | 21.30              |

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode) 

To install the dependency, please run
```shell
pip install -r requirements.txt
```

## Train and Validation

### Train

``` shell
python train.py --model=mobilenet_v3_large_100 --dataset=your_data_path --val_while_train --val_split=val --val_interval=1 --ckpt_save_dir your_save_path
```

You can add more parameters in the configs file by reading the paper, such as RandAugment etc.

### Validation

```python
python validate.py --model=mobilenet_v3_large_100 --dataset=your_data_path --val_split=validation --ckpt_path='./ckpt/mobilenet_v3-best.ckpt' 
``` 

### Acknowledgement

This work is sponsored by Natural Science Foundation of China(62276242), CAAI-Huawei MindSpore Open
Fund(CAAIXSJLJJ-2021-016B), Anhui Province Key Research and Development Program(202104a05020007), and
USTC Research Funds of the Double First-Class Initiative(YD2350002001)

### Citation

If you find this project useful in your research, please consider citing:

```latex
@article{yu2022scene,
  title={Scene Clustering Based Pseudo-labeling Strategy for Multi-modal Aerial View Object Classification},
  author={Yu, Jun and Chang, Hao and Lu, Keda and Zhang, Liwen and Du, Shenshen},
  journal={arXiv preprint arXiv:2205.01920},
  year={2022}
}
```
