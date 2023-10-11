## Integrating Image and Gene-Data with a Semi-Supervised Attention Model for Prediction of KRAS Gene Mutation Status in Non-Small Cell Lung Cancer

## 1.introduction

This paper proposes a Semi-supervised Multi-modal Multi-scale Attention Model (S2MMAM) to use medical image CT data and genetic data to effectively improve the accuracy of predicting KRAS gene mutation status in NSCLC.

## 2.Requirements

Install Pytorch 1.1.0 and CUDA 9.0

## 3. Data Preparation

* Download https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics<br/>
* Put the data under `./data/`

## 4.Train

* cd `scripts_lung` 
* Run `sh train.sh` to start the training process

## 5.Acknowledgement

Some code is reused from the [Pytorch implementation of mean teacher](https://github.com/CuriousAI/mean-teacher). 
