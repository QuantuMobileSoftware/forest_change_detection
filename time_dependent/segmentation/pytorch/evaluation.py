import os
import argparse

import imageio
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, auc, precision_recall_curve

from models.polyeval import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        required=True, help='Path to the directory all the data')
    parser.add_argument(
        '--prediction_path', '-pp', dest='prediction_path',
        required=True, help='Path to the directory with predictions')
    parser.add_argument(
        '--test_df_path', '-tp', dest='test_df_path',
        required=True, help='Path to the test dataframe with image names')
    parser.add_argument(
        '--output_name', '-on', dest='output_name',
        required=True, help='Name for output file')
    parser.add_argument(
        '--images_folder', '-imf', dest='images_folder',
        default='images',
        help='Name of folder where images are storing'
    )
    parser.add_argument(
        '--image_size', '-ims', default=56, 
        type=int, help='Image size'
    )
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--instances_folder', '-inf', dest='instances_folder',
        default='instance_masks',
        help='Name of folder where instances are storing'
    )
    parser.add_argument(
        '--image_type', '-imt', dest='image_type',
        default='tiff',
        help='Type of image file'
    )
    parser.add_argument(
        '--mask_type', '-mt', dest='mask_type',
        default='png',
        help='Type of mask file'
    )
    parser.add_argument(
        '--threshold', '-t', dest='threshold',
        default=0.35, type=float,
        help='Threshold for prediction'
    )

    return parser.parse_args()


def dice_coef(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f)+eps)


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

def confusion_matrix(y_true, y_pred):
    mm, mn, nm, nn = 0,0,0,0
    M, N = 0,0
    for i in range(len(y_true)):
        if(y_true.iloc[i]==y_pred.iloc[i]):
            if(y_true.iloc[i]==1):
                M+=1
                mm+=1
            else:
                N+=1
                nn+=1
        else:
            if(y_true.iloc[i]==1):
                M+=1
                mn+=1
            else:
                N+=1
                nm+=1
    return mm, mn, nm, nn, M, N

def evaluate(
        datasets_path, predictions_path, test_df_path, output_name, threshold,
        images_folder, image_type, masks_folder, mask_type, instances_folder, image_size
):

    res_cols = ['dataset_folder', 'name', 'position',
                'img_height', 'img_width', 'dice_score', 'iou_score',
                'pixel_amount']

    test_df_results = pd.DataFrame(columns=res_cols)

    filenames = pd.read_csv(test_df_path)

    dices, ious = [], []

    test_polys, truth_polys = [], []
    for ind, image_info in tqdm(filenames.iterrows()):

        name = '_'.join([image_info['name'], image_info['position']])

        prediction = cv.imread(f'{os.path.join(predictions_path, name)}.png')

        mask = cv.imread(os.path.join(
            datasets_path, image_info['dataset_folder'],
            masks_folder, f'{name}.{mask_type}'
        ))

        #print(prediction.shape)
        #print(mask.shape)

        test_polys.append(polygonize(prediction[:,:,0].astype(np.uint8)))
        truth_polys.append(polygonize(mask[:,:,0].astype(np.uint8)))

        dice_score = dice_coef(mask / 255, (prediction / 255) > threshold)
        iou_score = iou(mask / 255, (prediction / 255) > threshold, smooth=1.0)
        dices.append(dice_score)
        ious.append(iou_score)

        pixel_amount = mask.sum() / 255


        test_df_results = test_df_results.append({'dataset_folder': image_info['dataset_folder'] , 'name': name, 'position': image_info['position'],
                                'img_height': image_size, 'img_width': image_size,
                                'dice_score': dice_score, 'iou_score': iou_score, 'pixel_amount': pixel_amount}, ignore_index=True)

    # print("Metrics value - {0}".format(round(np.average(metrics), 4)))
    print("Average dice score - {0}".format(round(np.average(dices), 4)))
    print("Average iou  score - {0}".format(round(np.average(ious), 4)))

    '''
    count=0
    for tp in test_polys:
        if len(tp)==0: count+=1
        else: count+=len(tp)
    print(count)
    count=0
    
    for tp in truth_polys:
        if len(tp)==0: count+=1
        else: count+=len(tp)
    print(count)
    '''

    log_save = predictions_path + f'{output_name}_f1score.csv'
    log = pd.DataFrame(columns=['f1_score','threshold','TP','FP','FN'])
    for threshold in np.arange(0.1, 1, 0.1):
        F1score, true_pos_count, false_pos_count, false_neg_count, total_count = evalfunction(test_polys, truth_polys, threshold=threshold)
        log = log.append({'f1_score': round(F1score,4),
                            'threshold': round(threshold,2),
                            'TP':int(true_pos_count),
                            'FP':int(false_pos_count),
                            'FN':int(false_neg_count)}, ignore_index=True)
    
    print(log)
    log.to_csv(log_save, index=False)

    #F1score, true_pos_count, false_pos_count, false_neg_count, total_count = evalfunction(test_polys, truth_polys, threshold=0.5)
    #print(F1score, true_pos_count, false_pos_count, false_neg_count, total_count)
    '''
    if test_df_results['pixel_amount'].min()==0:
        y_true = (test_df_results['pixel_amount']>0)
        y_pred = (test_df_results['iou_score']>0.2)
        mm, mn, nm, nn, M, N = confusion_matrix(y_true, y_pred)
        print(f'M:{M},N:{N}')
        print(f'MM:{mm}, MN:{mn}, NM:{nm}, NN:{nn}')
        
        print(f"F1-score  - {round(f1_score( (test_df_results['pixel_amount']>0), (test_df_results['iou_score']>0.2)  ),4)}")
        print(f"precision - {round(precision_score( (test_df_results['pixel_amount']>0), (test_df_results['iou_score']>0.2)  ),4)}")
        print(f"recall - {round(recall_score( (test_df_results['pixel_amount']>0), (test_df_results['iou_score']>0.2)  ),4)}")
        #f1_score, precision, recall
        precision, recall, _ = precision_recall_curve( (test_df_results['pixel_amount']>0)*1, test_df_results['iou_score'])
        plt.plot(recall, precision, marker='.', label=output_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(predictions_path + f'{output_name}_PRC.png')
        plt.close()
    '''


    test_df_results_path = predictions_path + f'{output_name}_results.csv'
    test_df_results.to_csv(test_df_results_path, index=False)



if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.datasets_path, args.prediction_path,
        args.test_df_path, args.output_name,
        args.threshold, args.images_folder,
        args.image_type, args.masks_folder,
        args.mask_type, args.instances_folder,
        args.image_size
    )
