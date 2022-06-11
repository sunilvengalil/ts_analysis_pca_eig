from typing import Dict
from numpy import ndarray
from collections import defaultdict
from matplotlib import pyplot as plt

from utils import compute_svd, process_time_series_all, compute_pca_on_embedded_vecors
import os
import numpy as np
tau_array = [20]
M_array = [2, 4, 6,8, 10, 20]
max_num_samples_to_take = 5000
MIN_EIGEN_VALUE = 10

file_names = ["Delta", "Theta", "Beta", "Kai", "Lorenz", "Lordata", "Lambda", "Kappa", "Phi", "Gamma", "Alpha", "Mu", "Nu", "Rho"]
# file_names =["Mu", "Nu","Alpha", "Delta"]
ground_truths = ["STOCHASTIC", "NON-ST", "NON-ST", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST", "STOCHASTIC", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST"]
gt_dictionary = {file_name:ground_truth for file_name, ground_truth in zip(file_names, ground_truths) }

data_folder_or_single_file_name = "time_series_all"
shuffle = False

time_series = "all_series"
if shuffle:
    svd_results_folder = f"pca_on_embedded_vectors{time_series}_m_{M_array[0]}_{M_array[-1]}_tau_{tau_array[0]}_{tau_array[-1]}_shuffled_v2"
else:
    svd_results_folder = f"pca_on_embedded_vectors{time_series}_m_{M_array[0]}_{M_array[-1]}_tau_{tau_array[0]}_{tau_array[-1]}_v2"

if not os.path.isdir(svd_results_folder):
    os.mkdir(svd_results_folder)

if os.path.isfile(data_folder_or_single_file_name):
    files = [data_folder_or_single_file_name]
else:
    files = os.listdir(data_folder_or_single_file_name)

"""
Read data in to `data_dict`
Also create a dictionary for maximum value in the time series
"""
data_dict:Dict[str, ndarray] = dict(); # Dictionary of data array format {filename:data}
max_value_dict:Dict[str, float] = dict() # Dictionary of maximum values {filename:max_value}
print(files)
for file in files:
    print("Reading data from file", file)
    with open(data_folder_or_single_file_name + "/" + file, "r") as fp:
        text = fp.readlines()
    ts = np.asarray([float(d) for d in text])
    if shuffle:
        np.random.shuffle(ts)
    data_dict[file] = ts
    max_value = len(data_dict[file])
    max_value_dict[file] = max_value
    print(f"Number of samples in {file} {max_value}")

l = [k for k in data_dict.keys()]
print(l)

values = defaultdict(list)
import matplotlib

matplotlib.rc('axes', labelsize=25, )
matplotlib.rc('xtick', labelsize=25, )
matplotlib.rc('ytick', labelsize=25, )

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
# matplotlib.rc('font', **font)

for file_name in data_dict.keys():
    for M in M_array:
        for tau in tau_array:
            # results, num_splits, num_iters = process_time_series_all(np.asarray(data_dict[data_folder_or_single_file_name]),
            #                                                          min_eig_value=MIN_EIGEN_VALUE)
            # plot_eigen_ratio(data_dict[file_name], results, max_value_dict[file_name],
            #                  f"{pca_results_folder}/{file_name}.jpg",
            #                  title=f"{file_name}: Eigen ratio greater than {MIN_EIGEN_VALUE}")
            eig = compute_pca_on_embedded_vecors(data_dict[file_name], M, tau)
            plt.figure()
            plt.stem(eig)
            plt.xlabel("Principle Component Number", weight="bold")
            plt.ylabel("Eigen Values", weight="bold")
            plt.savefig(f"{svd_results_folder}/{file_name}_e1_vs_e2_m_{M}_tau_{tau}.jpg", bbox_inches="tight")

