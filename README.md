# SPGCD-Net: Semantic Prior Guided Change Detection Leveraging Pre-trained Vision-Language Models for Remote Sensing Imagery

## Dataset Preparation

SPGCD-Net is evaluated on LEVIR-CD, SYSU-CD, and WHU-CD, with processed dataset downloads and official source links provided below.

The repository includes the SPGCD-Net model implementation (`model/`), dataset loading (`CDDataset.py`), testing and evaluation scripts (`test.py`), and supplementary resources (`assets/`).

### 1. Download Processed Datasets

The processed datasets used for training, validation, and testing are available through the following Google Drive links:

- **LEVIR-CD:** [Download processed LEVIR-CD](https://drive.google.com/file/d/1t37PJyKth0THmGaywz24L-Qnu3Iv6enn/view?usp=drive_link)
- **SYSU-CD:** [Download processed SYSU-CD](https://drive.google.com/file/d/1ztTD8excmschSRDTdOBCVdUyDPEDFElT/view?usp=drive_link)
- **WHU-CD:** [Download processed WHU-CD](https://drive.google.com/file/d/1pkA4A69-Exabvb1UxwuFLweljEURbwuu/view?usp=drive_link)

### 2. Data Structure

Each processed dataset follows the directory structure below:

```text
dataset/
├── A/
├── B/
├── label/
```

- **`A/`**: Images acquired at time \(t_1\).
- **`B/`**: Images acquired at time \(t_2\), spatially corresponding to the images in `A/`.
- **`label/`**: Pixel-level binary change maps indicating changed and unchanged pixels.

Images in `A/` and `B/` are paired by filename, and each image pair is associated with its corresponding label map.

### 3. Official Dataset Sources

The original datasets can be accessed through their respective official sources:

- **LEVIR-CD:** [Official dataset website](https://justchenhao.github.io/LEVIR/)
- **SYSU-CD:** [Official dataset repository](https://github.com/liumency/SYSU-CD)
- **WHU-CD:** [Official dataset website](http://gpcv.whu.edu.cn/data/building_dataset.html)

## License

Original research materials associated with this study are made available under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0), which permits sharing with appropriate attribution for non-commercial purposes, provided that no adaptations are distributed.

CC BY-NC-ND 4.0: https://creativecommons.org/licenses/by-nc-nd/4.0/

Third-party datasets remain subject to their respective licenses and terms of use.
