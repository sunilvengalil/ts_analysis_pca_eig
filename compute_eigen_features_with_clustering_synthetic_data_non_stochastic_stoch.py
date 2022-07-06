import statistics
from collections import defaultdict
from typing import Dict

import numpy as np
import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

from utils import compute_svd, process_time_series_all, plot_eigen_ratio, plot_time_series, area_per_unit_length, \
    verify_tsne, \
    scatter_plot_2d, compute_pca_on_embedded_vecors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# MIN_EIGEN_VALUE = 10
# parser = argparse.ArgumentParser(description='Usage python3 compute_eigen_features.py --args.')
# parser.add_argument("--M", type=int, default=10,
#                    help='Number of dimensions')

# parser.add_argument('--tau', type=int, action='store_const',
#                    const=sum, default=200,
#                    help='Time delay of successive rows')

# parser.add_argument('--data_file', type=str,
#                    help='Path for folder where datafiles are located.Can also be a filename in case of single time_series file')
# args = parser.parse_args()
# print(args)

# file_names = ["Delta", "Sac_ascf_theta", "Sac_ascf_beta", "Sac_ascf_kai", "Lorenz", "Lordata", "Sac_ascf_lambda", "Kappa", "Sac_ascf_phi", "Sac_ascf_gamma", "Sac_Ascf_Alpha", "Sac_ascf_mu", "Sac_ascf_nu", "Sac_ascf_rho"]

# file_names = [f"beta_{beta}.000000_random_state_0",
#               f"beta_{beta}.000000_random_state_1",
#               f"beta_{beta}.000000_random_state_2",
#               f"beta_{beta}.000000_random_state_3",
#               f"beta_{beta}.000000_random_state_4",
#               f"beta_{beta}.000000_random_state_5"
#               ]

# file_name = f"compare_lorenz_with_linear_map"

file_name = f"non_stochastic_linear_map_stoch"
M = 10
tau = 20
shuffle = False

svd_results_folder = f"svdresults_synthetic_data"
pca_results_folder = f"pca_results_synthetic_data"

graphs_folder = f"{pca_results_folder}/graphs/"

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

min_eigen_values = list(range(5, 11))

# min_eigen_values = [10]
"""
Read Data
"""

LABEL_NON_STOCHASTIC = "NON-STOCHASTIC"
LABEL_STOCHASTIC = "STOCHASTIC"
ground_truths =[]
data_dict = {}
max_value_dict = {}
for file in files:
    print(file)
    if file.startswith("l") or file.startswith("a") or file.startswith("x") :
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

"""
Create the GT map
"""

gt_dictionary = {file_name.rsplit(".", 1)[0]:ground_truth for file_name, ground_truth in zip(files, ground_truths) }
# min_eigen_values = [2]
values = defaultdict(list)
scores = defaultdict(list)
min_eig_ratio_dict: Dict[str, Dict[int, int]] = defaultdict(str)  # {timeseries_name, {threshold:value}}
max_eig_ratio_dict: Dict[str, Dict[int, int]] = defaultdict(str)  # {threshold:value}
var_eig_ratio_dict: Dict[str, Dict[int, int]] = defaultdict(str)  # {threshold:value}
normalized_area_dict: Dict[int, int] = dict()  # {threshold:value}

i = 0
ground_truth = [None] * len(files)
print(gt_dictionary)
for file_name in data_dict.keys():
    key = file_name.rsplit(".",  1)[0]
    ground_truth[i] = gt_dictionary[key]
    i += 1

distinct_labels = [LABEL_STOCHASTIC, LABEL_NON_STOCHASTIC]

labels = [1] * len(files)
for i, gt in enumerate(ground_truth):
    if gt == LABEL_STOCHASTIC:
        labels[i] = 1
    else:
        labels[i] = 2

"""
Compute PCA Features
"""
for min_eigen_threshold_value in min_eigen_values:
    print("min_eigen_threshold_value", min_eigen_threshold_value)
    for file_name in data_dict.keys():
        print(f"{graphs_folder}/{file_name}.jpg")
        results, num_splits, num_iters = process_time_series_all(
            np.asarray(data_dict[file_name]),
            min_eig_value = min_eigen_threshold_value)

        image_filename = file_name.rsplit(".",1)[0] +".jpg"
        ts_filename = file_name.rsplit(".",1)[0] +"_ts.jpg"

        plot_eigen_ratio(data_dict[file_name],
                         eigenvalue_ratios=results,
                         fname=f"{graphs_folder}/{image_filename}",
                         num_samples=10000,
                         title=f"Max Eigen Ratio:{file_name}")
        plot_time_series(data_dict[file_name],
                         fname=f"{graphs_folder}/{ts_filename}",
                         title=f"Time Series:{file_name} : sample values"
                         )

        # insert into dictionary for creating dataframe
        key = file_name.rsplit(".", 1)[0]
        min_eig_ratio_dict[key]   = {min_eigen_threshold_value : min([r[0] for r in results])}
        max_eig_ratio_dict[key]   = {min_eigen_threshold_value : max([r[0] for r in results])}
        var_eig_ratio_dict[key]   = {min_eigen_threshold_value : statistics.variance([r[0] for r in results])}
        normalized_area_dict[key] = {min_eigen_threshold_value : area_per_unit_length(results, max_value)}

        # Plot scatter plots
        features = ["VER", "AUER"]
        # insert into dictionary for creating dataframe
        values["Time Series"].append(key)
        values["Ground Truth"].append(gt_dictionary[key])
        values["Max Eigen Ratio"].append(max_eig_ratio_dict[key][min_eigen_threshold_value])
        values["Variance of Eigen Ratio"].append(var_eig_ratio_dict[key][min_eigen_threshold_value])
        values["Normalized Area Under Eigen Ratio"].append(normalized_area_dict[key][min_eigen_threshold_value])
        values["Min Eigen Threshold Value"].append(min_eigen_threshold_value)

#       eig = compute_pca_on_embedded_vecors(data_dict[file_name], M, tau)
 #      values["Eigen Ratio on Embedded Vectors"].append(eig[0] / eig[-1])

    # Perform k-means clustering
    data = np.zeros([len(list(data_dict.keys())), 2])
    i = 0
    for key in var_eig_ratio_dict.keys():
        data[i][0] = var_eig_ratio_dict[key][min_eigen_threshold_value]
        data[i][1] = normalized_area_dict[key][min_eigen_threshold_value]
        i += 1

    scatter_plot_2d(data,
                    labels,
                    pca_results_folder,
                    f"variance_area_threshold_{min_eigen_threshold_value}",
                    #title=f"Variance and  Area Eigen Ratio Threshold = {min_eigen_threshold_value}",
                    legends= distinct_labels,
                    axis_labels=features,
                    log_axis=True
                    )
    feature_df  = pd.DataFrame(values)
    feature_df.to_csv(f"{pca_results_folder}/features.csv", index=False)

    model = KMeans(2)
    model.fit(data)
    scores["Threshold"].append(min_eigen_threshold_value)
    scores["K means silhouette score"].append(silhouette_score(data, model.labels_, metric="euclidean") )
#
df = pd.DataFrame(scores)
df.to_csv(f"{pca_results_folder}/threshold_vs_silhoutte_score.csv", index=False)

plt.figure()
plt.plot(scores["Threshold"], scores["K means silhouette score"])
plt.xlabel("Eigen Ratio threshold")
plt.ylabel("K means silhouette score")
plt.savefig(f"{pca_results_folder}/threshold_vs_silhoutte_score.jpg")

"""SVD computation"""
for file_name in data_dict.keys():
    svd = compute_svd(data_dict[file_name], M, tau)
    v_mat = svd[2]
    plt.figure()
    plt.plot(v_mat[0], v_mat[1])
    plt.xlabel("E1")
    plt.ylabel("E2")
    plt.savefig(f"{svd_results_folder}/{file_name}_e1_vs_e2_m_{M}_tau_{tau}.jpg")
