# mutual_info_img_txt

Joint learning of images and text via maximization of mutual information.

This repository incorporates the algorithms presented in <br />
Ruizhi Liao, Daniel Moyer, Miriam Cha, Keegan Quigley, Seth Berkowitz, Steven Horng, Polina Golland, William M Wells. [Multimodal Representation Learning via Maximization of Local Mutual Information.](https://arxiv.org/pdf/2103.04537.pdf) *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2021. <br />

This repo is a work-in-progress. As of now, we have released the code for joint representation learning of images and text by maximizing the mutual information between the feature embeddings of the two modalities. We demonstrate its application in learning from chest radiographs and radiology reports.


# Instructions

## Conda environment

Set up the conda environment using [`conda_environment.yml`](https://github.com/RayRuizhiLiao/mutual_info_img_txt/blob/main/conda_environment.yml):
```
conda env create -f conda_environment.yml
```

## BERT

Download the pre-trained BERT model, tokenizer, etc. from [`Dropbox`](https://www.dropbox.com/sh/snp8lr2afsgeb04/AACWNzsHSWksJGIWgp6P_T4ca?dl=0). You should download the folder *bert_pretrain_all_notes_150000* that contains seven files. The path to *bert_pretrain_all_notes_150000* should be passed to [`--bert_pretrained_dir`](https://github.com/RayRuizhiLiao/mutual_info_img_txt/blob/80d0c32e3625ef545cf2135beb0108847c113e4c/train_img_txt.py#L26).


## Model training

Train the model in an unsupervised fashion, i.e., optimizing [Eq (2)](https://arxiv.org/pdf/2103.04537.pdf):

```
python train_img_txt.py
```

When you run model training for the first time, it may take a while to tokenize the text. Afterwards, this process won't be repeated and the tokenized data will be saved for reuse. 


# Notes on Data

## MIMIC-CXR

We have experimented this algorithm on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), which is a large publicly available dataset of chest x-ray images with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA.

## Example data

We provide 16 example image-text pairs to test the code, listed in [`training_chexpert_mini.csv`](https://github.com/RayRuizhiLiao/mutual_info_img_txt/blob/main/example_data/training_chexpert_mini.csv).


# Contact

Ruizhi (Ray) Liao: ruizhi [at] mit.edu
