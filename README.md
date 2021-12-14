# mutual_info_img_txt

Joint learning of images and text via maximization of mutual information.

This repository incorporates the algorithms presented in <br />
Ruizhi Liao, Daniel Moyer, Miriam Cha, Keegan Quigley, Seth Berkowitz, Steven Horng, Polina Golland, William M Wells. [Multimodal Representation Learning via Maximization of Local Mutual Information.](https://arxiv.org/pdf/2103.04537.pdf) *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2021. <br />

This repo is a work-in-progress. As of now, we have released the code for joint representation learning of images and text by maximizing the mutual information between the feature embeddings of the two modalities. We demonstrate its application in learning from chest radiographs and radiology reports.


# Instructions

## Setup

Set up the conda environment using [`conda_environment.yml`](https://github.com/RayRuizhiLiao/joint_chestxray/blob/master/conda_environment.yml). You might want to remove the pip dependencies if that is creating an issue for you. You can manually install the spacy and scispacy dependencies using `pip install spacy` and `pip install scispacy`. Read [`https://allenai.github.io/scispacy/`](https://allenai.github.io/scispacy/) for more information on scispacy. 

## Training

Train the model in an unsupervised fashion, i.e., only the first term in [Eq (2)](https://arxiv.org/pdf/2103.04537.pdf) is optimized:

```
python train_img_txt.py
```

# Notes on Data

## MIMIC-CXR

We have experimented this algorithm on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), which is a large publicly available dataset of chest x-ray images with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA.

# Contact

Ruizhi (Ray) Liao: ruizhi [at] mit.edu
