# DTERN
Official PyTorch implementation of paper: Dual-Temporal Exemplar Representation Network for Video Semantic Segmentation

## Abstract
Video semantic segmentation aims to assign a class label for each pixel in every video frame. Existing methods predominantly follow the reference-target interaction paradigm, focusing on extracting local temporal context while neglecting the integration of global temporal information. Moreover, complex dynamics and varying lighting conditions introduce inter-frame intra-class discrepancies in feature representations, leading to unstable predictions. To tackle these issues, we propose a novel framework, the Dual-Temporal Exemplar Representation Network (DTERN), which utilizes the strong representational capability of cluster centers to effectively model both local and global temporal information. DTERN consists of two core modules: 1) the Local Temporal Exemplar Module (LTEM), which constructs local exemplars to capture local temporal context, ensuring stable and reliable predictions. 2) the Global Temporal Exemplar Module (GTEM), which introduces learnable global exemplars to dynamically model global temporal information, thereby improving segmentation consistency. Furthermore, we observe that the existing Video Consistency (VC) fails to evaluate segmentation accuracy and lacks sensitivity to small-object segmentation. To this end, we propose Video Effective Consistency (VEC) to comprehensively evaluate the temporal consistency and segmentation effectiveness. Experiments on VSPW and Cityscape demonstrate that DTERN outperforms state-of-the-art methods.

A portion of the code is currently under organization and will be made available shortly.


