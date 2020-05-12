import os
import cv2
import numpy as np

from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt

def watershed_segmentation(image):
	distance = ndi.distance_transform_edt(image)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
	                            labels=image)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=image)
	return labels, distance

def polygonize(raster_array, meta=None, transform=False):
    contours, hierarchy = cv2.findContours(raster_array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i in range(len(contours)):
        c = contours[i]
        n_s = (c.shape[0], c.shape[2])
        if n_s[0] > 2:
            if transform:
                polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
            else:
                polys = [tuple(i) for i in c.reshape(n_s)]
            polygons.append(Polygon(polys))
    return polygons

def iou_poly(test_poly, truth_poly):
    iou_score = 0
    intersection_result = test_poly.intersection(truth_poly)
    if not intersection_result.is_valid:
        intersection_result = intersection_result.buffer(0)
    if not intersection_result.is_empty:
        intersection_area = intersection_result.area
        union_area = test_poly.union(truth_poly).area
        iou_score = intersection_area / union_area
    else:
        iou_score = 0
    return iou_score


def score(test_polys, truth_polys, threshold=0.5):
    true_pos_count = 0
    true_neg_count = 0
    false_pos_count = 0
    false_neg_count = 0
    total_count = 0
    for test_poly, truth_poly in zip(test_polys, truth_polys):
        if len(test_poly)==0 and len(truth_poly)==0:
            true_neg_count += 1
            total_count+=1
        elif len(test_poly)==0 and len(truth_poly)>0:
            false_pos_count += 1
            total_count+=1
        elif len(test_poly)>0 and len(truth_poly)==0:
            false_neg_count += 1
            total_count+=1
        else:
            intersected=[]
            
            for test_p in test_poly:
                for truth_p in truth_poly:
                    if not test_p.is_valid:
                        test_p = test_p.buffer(0)
                    if not truth_p.is_valid:
                        truth_p = truth_p.buffer(0)
                    if test_p.intersection(truth_p).is_valid:
                        if not test_p.intersection(truth_p).is_empty:
                            intersected.append([test_p, truth_p])
                            
            if len(intersected) < len(test_poly):
                false_neg_count += (len(test_poly) - len(intersected))
                total_count+=(len(test_poly) - len(intersected))
            if len(intersected) < len(truth_poly):
                false_pos_count += (len(truth_poly) - len(intersected))
                total_count+=(len(truth_poly) - len(intersected))
            for inter in intersected:
                iou_score = iou_poly(inter[0], inter[1])

                '''
                xs, ys = inter[0].exterior.xy
                plt.fill(xs, ys, alpha=0.5, fc='r', label='Test')
                xs, ys = inter[1].exterior.xy
                plt.fill(xs, ys, alpha=0.5, fc='b', label='Truth')
                plt.legend()
                plt.title(iou_list)
                plt.show()
                '''

                if iou_score >= threshold:
                    true_pos_count += 1
                    total_count+=1
                else:
                    false_pos_count += 1
                    total_count+=1
            '''            
            for geom in test_poly:    
                xs, ys = geom.exterior.xy
                plt.fill(xs, ys, alpha=0.5, fc='r', label='Test')
            
            for geom in truth_poly:    
                xs, ys = geom.exterior.xy
                plt.fill(xs, ys, alpha=0.5, fc='b', label='Truth')
            plt.legend()
            plt.show()
            '''
    return true_pos_count, false_pos_count, false_neg_count, total_count


def evalfunction(test_polys, truth_polys, threshold = 0.5):
    if len(truth_polys)==0 and len(test_polys)!=0:
        true_pos_count = 0
        false_pos_count = len(test_polys)
        false_neg_count = 0
    elif len(truth_polys)==0 and len(test_polys)==0:
        true_pos_count = len(test_polys)
        false_pos_count = 0
        false_neg_count = 0
    else:
        true_pos_count, false_pos_count, false_neg_count, total_count = score(test_polys, truth_polys,
                                                                 threshold=threshold
                                                                 )

    if (true_pos_count > 0):
        precision = float(true_pos_count) / (float(true_pos_count) + float(false_pos_count))
        recall = float(true_pos_count) / (float(true_pos_count) + float(false_neg_count))
        F1score = 2.0 * precision * recall / (precision + recall)
    else:
        F1score = 0
    return F1score, true_pos_count, false_pos_count, false_neg_count, total_count
