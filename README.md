# DTERN
Official PyTorch implementation of paper: Dual-Temporal Exemplar Representation Network for Video Semantic Segmentation. (Accept by ICCV2025)

## Abstract
Video semantic segmentation aims to assign a class label for each pixel in every video frame. Existing methods predominantly follow the reference-target interaction paradigm, focusing on extracting local temporal contexts while neglecting the integration of global temporal information. Moreover, complex dynamics and varying lighting conditions introduce inter-frame intra-class discrepancies in feature representations, leading to unstable predictions. In this paper, we propose a novel framework, the Dual-Temporal Exemplar Representation Network (DTERN), which utilizes the strong representational capability of cluster centers, i.e., exemplars, to effectively model both local and global temporal information. DTERN consists of two core modules: 1) the Local Temporal Exemplar Module (LTEM), which constructs local exemplars to capture local temporal contexts, ensuring stable and reliable predictions. 2) the Global Temporal Exemplar Module (GTEM), which introduces learnable global exemplars to dynamically model global temporal information, thereby improving the effective consistency of segmentation. Furthermore, we observe that the existing Video Consistency (VC) metric fails to evaluate segmentation accuracy and lacks sensitivity to small-object segmentation. To this end, we propose Video Effective Consistency (VEC) to comprehensively evaluate temporal consistency and segmentation effectiveness. Experiments on VSPW and Cityscape demonstrate that DTERN outperforms state-of-the-art methods.

![block images](https://github.com/zlxilo/_DTERN/blob/main/overview.png)

## Installation
Please follow the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```timm==0.3.0, CUDA11.0, pytorch==1.7.1, torchvision==0.8.2, mmcv==1.3.0, opencv-python==4.5.2```

Download this repository and install by:
```
cd DTERN && pip install -e . --user
```
## Usage
### Data preparation
Please follow [VSPW](https://github.com/sssdddwww2/vspw_dataset_download) to download VSPW 480P dataset.
After correctly downloading, the file system is as follows:
```
vspw-480
├── video1
    ├── origin
        ├── .jpg
    └── mask
        └── .png
```
### Training
```
./tools/dist_train.sh local_configs/dtern/B1/dtern.b1.480x480.vspw2.160k.py 4 --work-dir model_path/vspw2/work_dirs_4g_b1
```
### [Weights](https://drive.google.com/drive/folders/1TGP32UjOXYA-UM12ljvwUTBqCT0Hzatg?usp=drive_link)
## License
This project is only for academic use. 

