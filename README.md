
# Forest Change Detection

## Description
This is a source code repository for DEEP LEARNING FOR REGULAR CHANGE DETECTION IN UKRAINIAN FOREST ECOSYSTEM WITH SENTINEL-2 (Kostiantyn Isaienkov, Mykhailo Yushchuk, Vladyslav Khramtsov, Oleg Seliverstov), 2020.

* Paper ([arXiv](https://arxiv.org/herewillbeourpaper), [Journal](https://arxiv.org/herewillbeourpaper))
* [Data](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing)
* [Deforestation monitoring system](http://bit.ly/clearcutq)

## Repository structure info
 * `baseline` - scripts for deforestation masks predictions with baseline models
 * `time-dependent` - scripts for forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models

## Setup
All dependencies are given in requirements.txt
Main setup configuration:
* python 3.6
* pytorch==1.4.0
* [catalyst](https://github.com/catalyst-team/catalyst)==19.05
* [segmentation_models](https://github.com/catalyst-team/catalyst)==0.1.0

Tested with Ubuntu + Nvidia GTX1060 with Cuda==10.2. 
CPU mode also should work, but not tested.

## Dataset
You can download our datasets directly from Google drive for the baseline and time-dependent models. The image tiles from Sentinel-2, which were used for our research, are listed in [`tiles` folder](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing).

The data include *.geojson polygons:
* [baseline](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing): 2318 polygons, **36UYA** and **36UXA**, **2016-2019** years;
* [time-dependent](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing): **36UYA** (two sets of separated annotations, 278 and 123 polygons -- for spring and summer seasons respectively, **2019** year) and **36UXA** (1404 polygons, **2017-2018** years).
The files contain the following columns: `tileID` (ID of a tile, which was annotated), `img_date` (the date, at which the tile was observed), and `geometry` (polygons of deforestation regions). 

Also, we provide the set of images and masks prepared for training segmentation models as the [Kaggle dataset](https://kaggledataset).

## Training
### Reproduce results
To reproduce the results, presented in our paper, run the pipeline (download data, prepare images, train the models), as described in README files in`baseline` and `time-dependent` folders.
### Training with new data
To train the models with the new data, you have to create train/valid/test (*.csv) files with specified location of images and masks, and make a minor changes in `Dataset` classes (for more information about location of these classes, see README files in `baseline` and `time-dependent` folders).

## Citation
If you use our code and/or dataset for your research, please cite our [paper](https://herewillbeourpaper):

K. Isaienkov, M. Yushchuk, V. Khramtsov, O. Seliverstov, Deep learning for regular change detection in Ukrainian forest ecosystem with Sentinel-2, 2020

## Questions
If you have questions after reading README, please email to [k.isaienkov@quantumobile.com](mailto:k.isaienkov@quantumobile.com).