# SPGCD-Net: Semantic Prior Guided Change Detection Leveraging Pre-trained Vision-Language Models for Remote Sensing Imagery

## Dataset Preparation

SPGCD-Net is evaluated on three publicly available remote sensing change detection datasets: LEVIR-CD, SYSU-CD, and WHU-CD. To facilitate reproducibility, we provide download links to the processed datasets used in our experiments, followed by the official sources of the original datasets.

The repository includes the SPGCD-Net model implementation (`model/`), dataset loading (`CDDataset.py`), testing and evaluation scripts (`test.py`), and supplementary resources (`assets/`).

### 1. Download Processed Datasets

The processed datasets used for training, validation, and testing are available through the following Google Drive links:

- **LEVIR-CD:** [Download processed LEVIR-CD](INSERT_LEVIR_GOOGLE_DRIVE_LINK)
- **SYSU-CD:** [Download processed SYSU-CD](INSERT_SYSU_GOOGLE_DRIVE_LINK)
- **WHU-CD:** [Download processed WHU-CD](INSERT_WHU_GOOGLE_DRIVE_LINK)

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

Original research materials associated with this study are made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0), which permits sharing and adaptation with appropriate attribution.

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Third-party datasets remain subject to their respective licenses and terms of use.
