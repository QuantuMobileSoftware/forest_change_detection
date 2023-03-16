# Unet-diff deforestation model inference

### Build image

For building image locally you <b>need</b> to place [model](https://drive.google.com/file/d/1rbJf9lPLScT4h-X3e0CAsOqMqZfA6KNI/view?usp=sharing) into  
`models/unet-diff.pth`

And also download `S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml`([link](https://sentinel.esa.int/documents/247904/1955685/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml)) file and place it into `data/landcovers/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml`

and then

`docker build -t quantumobile/deforestation .`


### Docker run command
```
docker run \
    --runtime nvidia \
    -e "AOI=POLYGON ((-85.299088 40.339368, -85.332047 40.241477, -85.134979 40.229427, -85.157639 40.34146, -85.299088 40.339368))" \
    -e "START_DATE=2020-05-01" \
    -e "END_DATE=2020-06-30" \
    -e "SENTINEL2_GOOGLE_API_KEY=/input/sentinel2_google_api_key.json" \
    -e "SENTINEL2_CACHE=/input/SENTINEL2_CACHE" \
    -e "OUTPUT_FOLDER=/output" \
    -v path_to_SENTINEL2_CACHE:/input/SENTINEL2_CACHE \
    -v path_to_sentinel2_google_api_key.json:/input/sentinel2_google_api_key.json \
    -v path_to_/output:/output \
    quantumobile/deforestation
```

Push to repository 
```
docker push quantumobile/deforestation
```
Pull image from repository
```
docker pull quantumobile/deforestation
```


# Forest Change Detection

## Description
This is a source code repository for DEEP LEARNING FOR REGULAR CHANGE DETECTION IN UKRAINIAN FOREST ECOSYSTEM WITH SENTINEL-2 (Kostiantyn Isaienkov, Mykhailo Yushchuk, Vladyslav Khramtsov, Oleg Seliverstov), 2020.

* Paper ([IEEE JSTARS Journal](https://ieeexplore.ieee.org/document/9241044))
* [Data](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing)
* [Deforestation monitoring system](http://bit.ly/clearcutq)
* [Weight of the best model, UNet-Diff](https://drive.google.com/file/d/1rbJf9lPLScT4h-X3e0CAsOqMqZfA6KNI/view?usp=sharing)

## Repository structure info
 * `baseline` - scripts for deforestation masks predictions with baseline models
 * `time-dependent` - scripts for forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models

## Setup
All dependencies are given in requirements.txt
Main setup configuration:
* python 3.6
* pytorch==1.4.0
* torchvision==0.5.0
* [catalyst](https://github.com/catalyst-team/catalyst)==19.05
* [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)==0.1.0

Tested with Ubuntu + Nvidia GTX1080ti with Cuda==10.1 and Nvidia GTX1060 with Cuda==10.2. 
CPU mode also should work, but not tested.

## Dataset
You can download our datasets directly from Google drive for the baseline and time-dependent models. The image tiles from Sentinel-2, which were used for our research, are listed in [`tiles` folder](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing).

The data include *.geojson polygons:
* [baseline](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing): 2318 polygons, **36UYA** and **36UXA**, **2016-2019** years;
* [time-dependent](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP?usp=sharing): **36UYA** (two sets of separated annotations, 278 and 123 polygons -- for spring and summer seasons respectively, **2019** year) and **36UXA** (1404 polygons, **2017-2018** years).
The files contain the following columns: `tileID` (ID of a tile, which was annotated), `img_date` (the date, at which the tile was observed), and `geometry` (polygons of deforestation regions).

## Training
### Reproduce results
To reproduce the results, presented in our paper, run the pipeline (download data, prepare images, train the models), as described in README files in`baseline` and `time-dependent` folders.
### Training with new data
To train the models with the new data, you have to create train/valid/test (*.csv) files with specified location of images and masks, and make a minor changes in `Dataset` classes (for more information about location of these classes, see README files in `baseline` and `time-dependent` folders).

## Citation
If you use our code and/or dataset for your research, please cite our [paper](https://ieeexplore.ieee.org/document/9241044):

```
@ARTICLE{Isaienkov2021,
  author={K. {Isaienkov} and M. {Yushchuk} and V. {Khramtsov} and O. {Seliverstov}},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2},
  year={2021},
  volume={14},
  number={},
  pages={364-376},
  doi={10.1109/JSTARS.2020.3034186}
}
```
