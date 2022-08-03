import statistics
from collections import defaultdict
from typing import Dict
import cv2
import numpy as np
import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

from utils import compute_svd, process_time_series_all, plot_eigen_ratio, plot_time_series, area_per_unit_length, \
    verify_tsne, \
    scatter_plot_2d, compute_pca_on_embedded_vecors, create_image_data
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


file_name = f"non_stochastic_linear_map_stoch"

M = 100
tau = 20
shuffle = False

svd_results_folder = f"svdresults_synthetic_data_wn_log_map_2Aug"
pca_results_folder = f"pca_results_synthetic_data_wn_log_map_2Aug"

graphs_folder = f"{pca_results_folder}/images_m_{M}/"

if not os.path.isdir(svd_results_folder):
    os.mkdir(svd_results_folder)

if not os.path.isdir(pca_results_folder):
    os.mkdir(pca_results_folder)

if not os.path.isdir(graphs_folder):
    os.mkdir(graphs_folder)

if os.path.isfile(file_name):
    files = [file_name]
else:
    files = os.listdir(file_name)

"""
Read Data
"""
LABEL_NON_STOCHASTIC = "NON-STOCHASTIC"
LABEL_STOCHASTIC = "STOCHASTIC"
distinct_labels = [LABEL_STOCHASTIC, LABEL_NON_STOCHASTIC]

def read_data():
    ground_truths = []
    data_dict = {}
    max_value_dict = {}
    for file in files:
        print(file)
        if file.startswith("l") or file.startswith("a") or file.startswith("x"):
            ground_truths.append(LABEL_NON_STOCHASTIC)
        else:
            ground_truths.append(LABEL_STOCHASTIC)

        with open(file_name + "/" + file, "r") as fp:
            text = fp.readlines()

        ts = np.asarray([float(d) for d in text])
        if shuffle:
            np.random.shuffle(ts)
        data_dict[file] = ts
        max_value = len(data_dict[file])
        max_value_dict[file] = max_value
        print(f"Number of samples in {file} {max_value}")
    return data_dict, ground_truths

data_dict, ground_truths = read_data()

"""
Create the GT map
"""
gt_dictionary = {file_name.rsplit(".", 1)[0]:ground_truth for file_name, ground_truth in zip(files, ground_truths) }

i = 0
ground_truth = [None] * len(files)
print(gt_dictionary)
for file_name in data_dict.keys():
    key = file_name.rsplit(".",  1)[0]
    ground_truth[i] = gt_dictionary[key]
    i += 1

labels = [1] * len(files)
for i, gt in enumerate(ground_truth):
    if gt == LABEL_STOCHASTIC:
        labels[i] = 1
    else:
        labels[i] = 2


"""Visualize data as image"""
for file_name in data_dict.keys():
    data_matrix = create_image_data(data_dict[file_name], M, tau)
    data_matrix = ((data_matrix - np.min(data_matrix)) / (np.max(data_matrix) - np.min(data_matrix))) * 256
    cv2.imwrite(f"{graphs_folder}/{file_name}.png",data_matrix)
