import argparse
import os

import cv2 as cv
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


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
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--instances_folder', '-inf', dest='instances_folder',
        default='masks', #'instance_masks',
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
        default=0.35,
        help='Threshold for prediction'
    )

    return parser.parse_args()


def watershed_transformation(prediction):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv.filter2D(prediction, cv.CV_32F, kernel)
    sharp = np.float32(prediction)
    imgResult = sharp - imgLaplacian

    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype(np.uint8)

    bw = cv.cvtColor(prediction, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    _, thresholded = cv.threshold(dist, 0.2, 1.0, cv.THRESH_BINARY)

    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dilated = cv.dilate(thresholded, kernel1)

    dist_8u = dilated.astype('uint8')
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    markers = np.zeros(dilated.shape, dtype=np.int32)

    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)

    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    cv.watershed(imgResult, markers)

    return markers


def post_processing(prediction):
    return watershed_transformation(prediction)


def dice_coef_logical(true_positives, false_positives, false_negatives):
    if true_positives + false_negatives + false_positives == 0:
        return 1
    return (2. * true_positives) / (2. * true_positives + false_positives + false_negatives)


def dice_coef(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f))


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)


def compute_iou_matrix(markers, instances):
    labels = np.unique(markers)

    labels = labels[labels < 255]
    labels = labels[labels > 0]
    iou_matrix = np.zeros((len(labels), len(instances)), dtype=np.float32)

    for i, label in enumerate(labels):
        prediction_instance = (markers == label).astype(np.uint8)

        for j, ground_truth_instance in enumerate(instances):
            iou_value = iou(prediction_instance, ground_truth_instance)
            iou_matrix[i, j] = iou_value

    return iou_matrix


def compute_metric_at_thresholds(iou_matrix):
    dices = []
    if iou_matrix.shape == (0, 0):
        return 1
    elif iou_matrix.shape[0] == 0:
        return 0
    for threshold in np.arange(0.5, 1, 0.05):
        true_positives = (iou_matrix.max(axis=1) > threshold).sum()
        false_positives = (iou_matrix.max(axis=1) <= threshold).sum()
        false_negatives = (iou_matrix.max(axis=0) <= threshold).sum()
        dices.append(dice_coef_logical(true_positives, false_positives, false_negatives))
    return np.average(dices)


def evaluate(
        datasets_path, predictions_path, test_df_path, output_name, threshold,
        images_folder, image_type, masks_folder, mask_type, instances_folder
):
    filenames = pd.read_csv(test_df_path)

    metrics = []

    writer = tf.python_io.TFRecordWriter(
        os.path.join(os.path.dirname(predictions_path), f'{output_name}.tfrecords'))

    dices = []

    for ind, image_info in tqdm(filenames.iterrows()):

        name = '_'.join([image_info['name'], image_info['position']])

        prediction = cv.imread(f'{os.path.join(predictions_path, name)}.png')

        image = imageio.imread(os.path.join(
            datasets_path, image_info['dataset_folder'],
            images_folder, f'{name}.{image_type}'
        ))[:, :, :3]

        mask = cv.imread(os.path.join(
            datasets_path, image_info['dataset_folder'],
            masks_folder, f'{name}.{mask_type}'
        ))

        img_size = image.shape
        instances = []
        '''
        image_instances_path = os.path.join(
            datasets_path, image_info['name'],
            instances_folder, name
        )

        for instance_name in os.listdir(image_instances_path):
            if ".png" in instance_name and ".xml" not in instance_name:
                rgb_instance = cv.imread(os.path.join(image_instances_path, instance_name))
                bw_instance = cv.cvtColor(rgb_instance, cv.COLOR_BGR2GRAY)
                instances.append(bw_instance)

        markers = post_processing(prediction)

        iou_matrix = compute_iou_matrix(markers, instances)
        metric = compute_metric_at_thresholds(iou_matrix)
        metrics.append(metric)
	'''
        img_raw = Image.fromarray(np.uint8(image), 'RGB').tobytes()
        msk_raw = Image.fromarray(np.uint8(mask), 'RGB').tobytes()
        pred_raw = Image.fromarray(np.uint8(prediction), 'RGB').tobytes()

        dice_score = dice_coef(mask / 255, (prediction / 255) > threshold)
        dices.append(dice_score)

        example = tf.train.Example(features=tf.train.Features(feature={
            "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[1])])),
            "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[0])])),
            "dice_score": tf.train.Feature(float_list=tf.train.FloatList(value=[dice_score])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "mask_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk_raw])),
            "pred_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pred_raw])),
            #"metric": tf.train.Feature(float_list=tf.train.FloatList(value=[metric])),
            "img_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
            "msk_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
            "pred_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
        }))
        writer.write(example.SerializeToString())

    # print("Metrics value - {0}".format(round(np.average(metrics), 4)))
    print("Average dice score - {0}".format(round(np.average(dices), 4)))


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.datasets_path, args.prediction_path,
        args.test_df_path, args.output_name,
        args.threshold, args.images_folder,
        args.image_type, args.masks_folder,
        args.mask_type, args.instances_folder
    )
