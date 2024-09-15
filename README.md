# KCMM    

This code base is the pytorch implementation of the paper:     
**Knowledge-Driven Compositional Action Recognition**   


## Requirements   

```
conda create -n KCMM python=3.8
conda activate KCMM
torch==1.12.0
torchvision==0.13.0
timm==1.0.9
transformers==2.5.1
```  

## Dataset  

Download Something-Something Dataset and Something-Else Annotation from [Something-Else repo](https://github.com/joaanna/something_else) (Joaana et al., 2020). 

Extract (or softlink) videos under `dataset/sth_else/videos`, and then dump the frames into `dataset/sth_else/frames` following [Interactive_Fusion_for_CAR](https://github.com/ruiyan1995/Interactive_Fusion_for_CAR)

## Pre-Checkpoints files  

Create a `pretrained_weights` directory in the `model` path.

First, download pretrained models such as `swin_tiny_patch244_window877_kinetics400_1k.pth` and `MViTv2_S_16x4_k400_f302660347.pyth` from the [Slowfast repository](https://github.com/facebookresearch/SlowFast/tree/main/projects/mvitv2).

Then download the `conceptnet5.zip` and `common_best_model.pth` models from the following [google drive](https://drive.google.com/drive/folders/1qIEn4WjVasI3rCrV81pa_G28NuRSB05e?usp=sharing), where `conceptnet5.zip` is unzipped to the `model` directory, and all other models are placed in the `model/pretrained_weights` directory

## Getting Started    

To train, test or conduct KCMM, please run `scripts/train.sh` or `scripts/test.sh`.

## Acknowledgments   

We used parts of code from following repositories:

https://github.com/joaanna/something_else

https://github.com/ruiyan1995/Interactive_Fusion_for_CAR 

https://github.com/pengzhansun/Counterfactual-Debiasing-Network  





