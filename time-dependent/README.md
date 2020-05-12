# Forest Change Detection: time-dependent models

Source code for "DEEP LEARNING FOR HIGH-FREQUENCY CHANGE DETECTION IN UKRAINIAN FOREST ECOSYSTEM WITH SENTINEL-2 (K. Isaienkov+, 2020) paper

## Project structure info
 * `data_prepare` - scripts for data download and preparation
 * `segmentation` - investigation about model approach, model training and model evaluation of clearcut detection

## Credential setup

This project needs several secure credentials, for peps.cnes.fr and sentinel-hub.
For correct setup, you need to create peps_download_config.ini (it could be done by example peps_download_config.ini.example) and feel auth, password and sentinel_id parameters.

## Model Development Guide
### Data downloading (`data_prepare` folder)

1) Create an account on https://peps.cnes.fr/rocket/#/home

2) Specify params in config file input/peps_download_config.ini : most valuable parameters are `start_date,end_date,tile`, which have to be set to exact values, and `latmin,latmax,lonmin,lonmax`, which could be specified approximately for a given tile.

3) Download an image archive with `python peps_download.py`.

4) Unzip the archive.

5) Merge bands with `python prepare_tif.py --data_folder … --save_path …` for a single image folder, or `./PREPARE_IMAGES.sh "data_folder" "save_path"` for the catalogue of images. 

6) Run `prepare_clouds.py` (by defaults, this script is executing with `./PREPARE_IMAGES.sh "data_folder" "save_path"` script). This scripts works as for Level-C images (detection clouds with `sentinel-cloud-detector` API), as well as for Level-A images (coping and resampling available cloud map from image archive).

The output of each script are `.tif` merged bands, `.png` image, each separately for each band, and `_clouds.tiff` images with clouds detections.

Also, data could be downloaded manually from https://scihub.copernicus.eu/dhus/#/home:

1) Create an account

2) Specify the tileID in search query (e.g., 36UYA)

3) Choose the wanted dates of imaging, and download with the arrow-shaped buttom (Download Product)

Keep in mind, that `scihub.copernicus.eu` allows to download 3 images maximum at once.

### Data preparation
1) Create folder (e.g., `data`, on the same level with `data_prepare` and `segmentation`) where the following datasets are stored:
   * Source subfolder stores raw data that has to be preprocess (`source`)
   * Input subfolder stores data that is used in training and evaluation (`input`)
   * Polygons subfolder stores markup (`polygons`)
   * Subfolder containing cloud maps for each image tile (`auxiliary`)

2) The source folder contains folders for each image that you downloaded. In that folder you have to store TIFF images of channels (in our case, 'rgb', 'b8', 'b8a', 'b10', 'b11', 'b12', 'ndvi', 'ndmi' channels) named as f”{image_folder}\_{channel}.tif”.

3) If you have already merged bands to a single TIFF, you can just move it to `input` folder. But you have to create the folder (it may be empty) for these images in the source folder.

4) The polygons folder contains markup that you apply to all images in input folder. Polygons could be as a single file, as well as different files (e.g., if you want to process each labeled series of tiles separately, to avoid collisions between different markups)

#### Example of data folder structure:
```
data
├── auxiliary
│   ├── image0_clouds.tiff
│   └── image1_clouds.tiff
├── input
│   ├── image0.tif
│   └── image1.tif
├── polygons
│   └── markup.geojson
└── source
    ├── image0
    │   ├── image0_b8.tif
    │   ├── image0_b8a.tif
    │   ├── ...
    │   └── image0_rgb.tif
    └── image1
        ├── iamge1_b8.tif
        ├── image1_b8a.tif
        ├── ...
        └── image1_rgb.tif
```
5) Run preprocessing on this data. You can specify other params if it necessary (**add --no_merge if you have already merged channels with prepare_tif.py script**).
```
python preprocessing.py \
 --polys_path ../data/polygons \
 --tiff_path ../data/source
 --save_path ../data/input
```
The output of this scipt are subfolders in the `input` path, which contain divided tile images \ tile masks into the small pieces, with specified `--width` and `--height`.

6) After preprocessing, run the script for dividing cloud maps into pieces (`python split_clouds.py`) with specified parameters.

#### Example of input folder structure after preprocessing:
```
input
├── image0
│   ├── geojson_polygons
│   ├── image0.png
│   ├── image_pieces.csv
│   ├── images
│   ├── masks
│   └── clouds
├── image0.tif
├── image1
│   ├── geojson_polygons
│   ├── image1.png
│   ├── image_pieces.csv
│   ├── images
│   ├── masks
│   └── clouds
└── image1.tif
```

6) Run image sequence creation script. Run the script which is related to the model in use; these scipts are separated in the following way:
```
python image_difference.py
```
- to create input images for UNet-diff and UNet-CH models (parameter --classification_head, which is `boolean` value, is corresponded for choice of the model). Output of the script are three images stacked together (img1, img2, and their difference) and difference mask;

```
python image_siamese.py
```
- to create input images for UNet2D, UNet3D, Siam-Conc and Siam-Diff models. Output of the script are separated files for each pair of images from the sequence and difference mask;

```
python image_lstm.py
```
- to create input images for UNet-LSTM model. Output of the script are separated images from the sequence (length of which may be specified in the script) and mask for each image from sequence;

These scripts create the folder at the specified location (by default, `../data/folder_name`), each folder consists of subfolders with sequence of images\masks. Each scripts creates the input files ready for learning, with spatial train/test/val datasets. The datasets with images\masks paths are in `.csv` files within the subfolders with sequence of images/masks.

### Models overview
## Implemented models:
  * UNet-diff (pytorch/models/utils.py `get_model('unet18')`)
  * UNet-CH   (pytorch/models/utils.py `get_model('unet18', classification_head=True)`)
  * UNet2D    (pytorch/models/siamese.py `Unet`)
  * UNet3D    (pytorch/models/siamese.py `Unet3D`)
  * Siam-Diff (pytorch/models/siamese.py `SiamUnet_diff`)
  * Siam-Conc (pytorch/models/siamese.py `SiamUnet_conc`)
  * UNet-LSTM (pytorch/models/u_lstm.py `Unet_LstmDecoder`)
  
1) If it necessary specify augmentation in pytorch/dataset.py for `Dataset`, `SiamDataset`, and `LstmDataset`.

2) Specify hyperparams in pytorch/train.py (for image difference), pytorch/trainsiam.py (for siamese networks; `Trainer` class is in pytorch/models/utils.py file), and pytorch/trainlstm.py (for LSTM network)

3) Run training `python train.py` (for UNet-diff and UNet-CH), or `python trainlstm.py` (from UNet-LSTM model), or `python trainsiam.py` (for UNet2D, UNet3D, and siamese models).

### Model training and evaluation
1) Train the models, generate predictions and evaluate the models with Dice and F1-scores:
* UNet-diff or UNet-CH models (the difference is in specified `head` parameter):
```
data_path=../data/diff
image_size=56
neighbours=3
epochs=200
lr=1e-2
image_size=56

network=unet18
loss=bce_dice
optimizer=Adam

head=True

name="diff_"$network"_"$optimizer"_"$loss"_"$lr"_"$head

python train.py --epochs $epochs \
                --image_size $image_size \
 			          --lr $lr \
                --network $network \
                --optimizer $optimizer \
                --loss $loss \
                --name $name \
                --dataset_path $data_path/ \
                --train_df $data_path/train_df.csv \
                --val_df $data_path/valid_df.csv \
                --channels rgb b8 b8a b11 b12 ndvi ndmi \
                --neighbours $neighbours \
                --classification_head $head

mkdir ../data/predictions/$name

echo "Test"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/test_df.csv --save_path ../data/predictions/$name --network $network --size $image_size

echo "Train"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/train_df.csv --save_path ../data/predictions/$name --network $network --size $image_size

echo "Valid"
python prediction.py --classification_head $head --neighbours 3 --channels rgb b8 b8a b11 b12 ndvi ndmi --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/valid_df.csv --save_path ../data/predictions/$name --network $network --size $image_size

cut=0.4
echo "Test"
python evaluation.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/test_df.csv --output_name 'test' --threshold $cut

echo "Train"
python evaluation.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/train_df.csv --output_name 'train'  --threshold $cut

echo "Valid"
python evaluation.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/valid_df.csv --output_name 'val'  --threshold $cut
```  
* UNet2D, UNet2D, Siam-Conc, Siam-Diff:
```
data_path=../data/siam

epochs=200
lr=1e-2
image_size=56
optimizer=Adam

loss=bce_dice
model=unet

name="siam"_$model"_"$optimizer"_"$loss"_"$lr

#train
python trainsiam.py --epochs $epochs \
                --image_size $image_size \
                --lr $lr \
                --model $model \
                --network unet18 \
                --optimizer $optimizer \
                --loss $loss \
                --name $name \
                --dataset_path $data_path/ \
                --train_df $data_path/train_df.csv \
                --val_df $data_path/onlymasksplit/valid_df.csv \
                --test_df $data_path/test_df.csv \
                --mode train

#train predict
python trainsiam.py --epochs $epochs \
                --image_size $image_size \
                --lr $lr \
                --model $model \
                --network unet18 \
                --optimizer $optimizer \
                --loss $loss \
                --name $name \
                --dataset_path $data_path/ \
                --train_df $data_path/train_df.csv \
                --val_df $data_path/valid_df.csv \
                --test_df $data_path/test_df.csv \
                --mode eval
```
* UNet-LSTM model (pay attention to the `--neighbours` parameter, it should be equal to the number of images in sequence, specified during preprocessing):
```
data_path=../data/lstm_diff

epochs=200
lr=1e-2
image_size=56

loss=tversky
optimizer=Adam

model=lstm_decoder

name=$model"_"$optimizer"_"$loss"_"$lr"_tmp2"

python trainlstm.py --epochs $epochs \
                --image_size $image_size \
                --lr $lr \
                --model $model \
                --optimizer $optimizer \
                --loss $loss \
                --name $name \
                --dataset_path $data_path/ \
                --train_df $data_path/train_df.csv \
                --val_df $data_path/valid_df.csv \
                --test_df $data_path/test_df.csv \
                --allmasks False \
                --neighbours 5

python prediction_lstm.py --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/train_df.csv --save_path ../data/predictions/$name/ --channels rgb b8 b8a b11 b12 ndvi ndmi --neighbours 5 --size $image_size
python prediction_lstm.py --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/valid_df.csv --save_path ../data/predictions/$name/ --channels rgb b8 b8a b11 b12 ndvi ndmi --neighbours 5 --size $image_size
python prediction_lstm.py --data_path $data_path --model_weights_path ../logs/$name/checkpoints/best.pth --test_df $data_path/test_df.csv --save_path ../data/predictions/$name/ --channels rgb b8 b8a b11 b12 ndvi ndmi --neighbours 5 --size $image_size

sample=train
echo "$sample"
python evaluation_lstm.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/$sample"_df.csv" --output_name $sample --threshold 0.4

sample=valid
echo "$sample"
python evaluation_lstm.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/$sample"_df.csv" --output_name $sample --threshold 0.4

sample=test
echo "$sample"
python evaluation_lstm.py --datasets_path $data_path --prediction_path ../data/predictions/$name/predictions --test_df_path $data_path/$sample"_df.csv" --output_name $sample --threshold 0.4
```
Implemented optimizers:
* Adam
* SGD
* RAdam

Implemented losses:
* BCE+Dice
* Lovasz
* Focal
* Tversky (alpha=beta=0.5 -> Dice losss)