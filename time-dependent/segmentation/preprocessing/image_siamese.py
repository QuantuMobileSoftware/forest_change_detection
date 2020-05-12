import os
import cv2
import csv
import random
import imageio
import datetime
import argparse

import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp
import matplotlib.pyplot as plt

from tqdm import tqdm
from random import random
from skimage import img_as_ubyte, io
from imgaug import augmenters as iaa
from scipy.ndimage import gaussian_filter
from skimage.transform import match_histograms
from rasterio.plot import reshape_as_image as rsimg

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for dividing images into smaller pieces.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data/input', help='Path to input data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/siam',
        help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        default='../data/polygons', 
        help='Path to the polygons'
    )
    parser.add_argument(
        '--img_path', '-ip', dest='img_path',
        default='images', help='Path to pieces of image'
    )
    parser.add_argument(
        '--msk_path', '-mp', dest='msk_path',
        default='masks', help='Path to pieces of mask'
    )
    parser.add_argument(
        '--cld_path', '-cp', dest='cld_path',
        default='clouds', help='Path to pieces of cloud map'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=56,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=56,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--neighbours', '-nbr', dest='neighbours', default=6,
        type=int, help='Number of pairs before the present day (dt=5 days, e.g. neighbours=3 means max dt = 15 days)'
    )
    parser.add_argument(
        '--train_size', '-tr', dest='train_size',
        default=0.7, type=float, help='Represent proportion of the dataset to include in the train split'
    )
    parser.add_argument(
        '--test_size', '-ts', dest='test_size',
        default=0.15, type=float, help='Represent proportion of the dataset to include in the test split'
    )
    parser.add_argument(
        '--valid_size', '-vl', dest='valid_size',
        default=0.15, type=float, help='Represent proportion of the dataset to include in the valid split'
    )
    return parser.parse_args()

def getdates(data_path):
    tiles = [ [name, datetime.datetime.strptime(name[-15:-11]+'-'+name[-11:-9]+'-'+name[-9:-7], 
        '%Y-%m-%d')] for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name)) ]
    return tiles

def readtiff(filename):
    src = rs.open(filename)
    return rsimg(src.read()), src.meta

def imgdiff(tile1, tile2, diff_path, save_path, data_path, img_path, msk_path, cloud_path, writer, width,height):
    xs = [piece.split('_')[4:5][0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    ys = [piece.split('_')[5:6][0].split('.')[0] for piece in os.listdir(os.path.join(data_path,tile1,img_path))]
    assert len(xs)==len(ys)
    for i in range(len(xs)):
        if os.path.exists(os.path.join(data_path,tile1,img_path,tile1+'_'+xs[i]+'_'+ys[i]+'.tiff')) and os.path.exists(os.path.join(data_path,tile2,img_path,tile2+'_'+xs[i]+'_'+ys[i]+'.tiff')):
            img1,meta = readtiff( 
                            os.path.join(data_path,tile1,img_path,tile1+'_'+xs[i]+'_'+ys[i]+'.tiff') )
            img2, _   = readtiff( 
                            os.path.join(data_path,tile2,img_path,tile2+'_'+xs[i]+'_'+ys[i]+'.tiff') ) 
            
            msk1=imageio.imread(
                                os.path.join(data_path,tile1,msk_path,tile1+'_'+xs[i]+'_'+ys[i]+'.png'))
            msk2=imageio.imread(
                                os.path.join(data_path,tile2,msk_path,tile2+'_'+xs[i]+'_'+ys[i]+'.png'))

            cld1=io.imread(
                          os.path.join(data_path,tile1,cloud_path,tile1+'_'+xs[i]+'_'+ys[i]+'.tiff'))
            cld2=io.imread(
                          os.path.join(data_path,tile2,cloud_path,tile2+'_'+xs[i]+'_'+ys[i]+'.tiff'))
        else:
            continue

        if np.max(cld1)<0.2 and np.max(cld2)<0.2:
            img2 = match_histograms(img2, img1, multichannel=True)
            diff_msk = np.abs(msk1-msk2)
            name = diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.png'

            diff_msk = (gaussian_filter(diff_msk , 0.5)>0)*255
            diff_msk = diff_msk.astype(np.uint8)
            diff_msk = cv2.resize(diff_msk, (height,width), interpolation = cv2.INTER_NEAREST)
		
            meta['width'] = width
            meta['height'] = height

            u=0
            for img in [img1,img2]:
                u+=1
                with rs.open(os.path.join(diff_path, img_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+f'_{u}.tiff'), 'w', **meta) as dst:
	                for ix in range(img.shape[2]):
	                    dst.write(img[:, :, ix], ix + 1)
                dst.close()

            imageio.imwrite(os.path.join(diff_path, msk_path, diff_path.split('/')[-1]+'_'+xs[i]+'_'+ys[i]+'.png'), diff_msk)
            writer.writerow([
                diff_path.split('/')[-1], diff_path.split('/')[-1], xs[i]+'_'+ys[i], int(diff_msk.sum()/255)
            ])
        else: pass

def get_diff_and_split(data_path, save_path, polys_path, img_path, msk_path, cloud_path, width, height, neighbours, train_size, test_size, valid_size):
    tiles=getdates(data_path)
    df = pd.DataFrame(tiles, columns=['tileID','img_date'])
    df = df.sort_values(['img_date'],ascending=False)
    
    infofile=os.path.join(save_path,'data_info.csv')

    markups = [gp.read_file(os.path.join(polys_path, shp)) for shp in os.listdir(polys_path)]
    for shp in markups:
        shp['img_date'] = shp['img_date'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
        )

    with open(infofile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'dataset_folder', 'name', 'position', 'mask_pxl'
		])
        for i in range(len(df)-1):
            for j in range(i+1, i+1+neighbours):
                if(j<len(df)):
                    print(str(df['img_date'].iloc[i].date())+' - '+str(df['img_date'].iloc[j].date()))
                    print(f"dt={(df['img_date'].iloc[i]-df['img_date'].iloc[j]).days} days")
                    diff_path = os.path.join(save_path, str(df['img_date'].iloc[i].date())+'_'+str(df['img_date'].iloc[j].date()))
                    markup_number_i,markup_number_j=0,0
                    for shp_num in range(len(markups)):
	                    if df['img_date'].iloc[i]>=markups[shp_num]['img_date'].min() and df['img_date'].iloc[i]<=markups[shp_num]['img_date'].max():
	                    	markup_number_i=shp_num
	                    if df['img_date'].iloc[j]>=markups[shp_num]['img_date'].min() and df['img_date'].iloc[j]<=markups[shp_num]['img_date'].max():
	                    	markup_number_j=shp_num

                    if (df['img_date'].iloc[i]-df['img_date'].iloc[j]).days > neighbours*5:
	                    pass
                    elif markup_number_i!=markup_number_j:
	                    pass
                    else:
	                    if not os.path.exists(diff_path):
	                        os.mkdir(diff_path)
	                    if not os.path.exists(os.path.join(diff_path,img_path)):
	                        os.mkdir(os.path.join(diff_path,img_path))
	                    if not os.path.exists(os.path.join(diff_path,msk_path)):
	                        os.mkdir(os.path.join(diff_path,msk_path))

	                    imgdiff(df['tileID'].iloc[i], df['tileID'].iloc[j],diff_path,save_path,
                        data_path, img_path, msk_path,cloud_path,
                        writer,width,height)

            
    df = pd.read_csv(infofile)
    xy = df['position'].unique()
    
    np.random.seed(seed=59)
    rand = np.random.random(size=len(xy))
    
    train=[]
    test=[]
    valid=[]
    for i in range(len(xy)):
        if rand[i]<=train_size:
            train.append(xy[i])
        elif rand[i]>train_size and rand[i]<train_size+test_size:
            test.append(xy[i])
        else:
            valid.append(xy[i])
    
    os.system(f'mkdir {save_path}/onlymasksplit')
    for data_type, name_type in zip([train,test,valid],['train','test','valid']):
        markups=0
        output_file=os.path.join(save_path,f'{name_type}_df.csv')
        os.system(f'head -n1 {infofile} > {output_file}')
        os.system(f'head -n1 {infofile} > {os.path.join(save_path,"onlymasksplit",f"{name_type}_df.csv")}')

        for position in data_type:
            df[df['position']==position].to_csv(output_file,mode='a',header=False,index=False,sep=',')
            df[(df['position']==position) & (df['mask_pxl']>0)].to_csv(os.path.join(save_path,'onlymasksplit',f'{name_type}_df.csv'),mode='a',header=False,index=False,sep=',')
            markups+=df[df['position']==position].shape[0]
        print(f"{name_type} markups: {markups}")

if __name__ == '__main__':
    args = parse_args()
    assert args.train_size + args.test_size + args.valid_size==1.0

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path,'img_msk_plots')):
        os.mkdir(os.path.join(args.save_path,'img_msk_plots'))

    get_diff_and_split(args.data_path, args.save_path, args.polys_path, args.img_path, args.msk_path, args.cld_path,
                       args.width,args.height, args.neighbours,
                       args.train_size, args.test_size, args.valid_size)