import statistics
from collections import defaultdict

import numpy as np
import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from utils import compute_svd, process_time_series_all, plot_eigen_ratio, area_per_unit_length, verify_tsne, \
    scatter_plot_2d
from sklearn.manifold import TSNE

MIN_EIGEN_VALUE = 10

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
file_names = ["Delta", "Theta", "Beta", "Kai", "Lorenz", "Lordata", "Lambda", "Kappa", "Phi", "Gamma", "Alpha", "Mu", "Nu", "Rho"]
ground_truths = ["STOCHASTIC", "NON-ST", "NON-ST", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST", "STOCHASTIC", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST"]
gt_dictionary = {file_name:ground_truth for file_name, ground_truth in zip(file_names, ground_truths) }


file_name = "time_series"
M = 2
tau = 20
shuffle = False


if shuffle:
    svd_results_folder = f"svd_results_m_{M}_tau_{tau}_shuffled_v2"
    pca_results_folder = f"pca_results_m_{M}_tau_{tau}_shuffled_v2"
else:
    svd_results_folder = f"svd_results_m_{M}_tau_{tau}_v2"
    pca_results_folder = f"pca_results_m_{M}_tau_{tau}_v2"

if not os.path.isdir(svd_results_folder):
    os.mkdir(svd_results_folder)

if not os.path.isdir(pca_results_folder):
    os.mkdir(pca_results_folder)

if os.path.isfile(file_name):
    files = [file_name]
else:
    files = os.listdir(file_name)


data_dict = {}
max_value_dict = {}
for file in files:
    with open(file_name + "/" + file, "r") as fp:
        text = fp.readlines()
    ts = np.asarray([float(d) for d in text])
    if shuffle:
        np.random.shuffle(ts)

    data_dict[file] = ts
    max_value = len(data_dict[file])
    max_value_dict[file] = max_value
    print(f"Number of samples in {file} {max_value}")

values = defaultdict(list)

for file_name in data_dict.keys():
    results, num_splits, num_iters = process_time_series_all(np.asarray(data_dict[file_name]),min_eig_value=MIN_EIGEN_VALUE)
    plot_eigen_ratio(data_dict[file_name], results, max_value_dict[file_name], f"{pca_results_folder}/{file_name}.jpg", title=f"{file_name}: Eigen ratio greater than {MIN_EIGEN_VALUE}")
    svd = compute_svd(data_dict[file_name], M, tau)
    v_mat = svd[2]
    plt.figure()
    plt.plot(v_mat[0], v_mat[1])
    plt.xlabel("E1")
    plt.ylabel("E2")
    plt.savefig(f"{svd_results_folder}/{file_name}_e1_vs_e2_m_{M}_tau_{tau}.jpg")

    min_eig_ratio = min([r[0] for r in results])
    max_eig_ratio = max([r[0] for r in results])
    var_eig_ratio = statistics.variance([r[0] for r in results])
    normalized_area = area_per_unit_length(results, max_value)

    key = file_name.rsplit(".",  1)[0]
    if key.startswith("sac_ascf_"):
        key = key.replace("sac_ascf_", "")

    # insert into dictionary for creating dataframe
    values["Timeseries"].append(key.capitalize())
    values["Ground Truth"].append(gt_dictionary[key.capitalize()])
    values["Max Eigen Ratio"].append(max_eig_ratio)
    values["Variance of Eigen Ratio"].append(var_eig_ratio)
    values["Normalized Area Under Eigen Ratio"].append(normalized_area)

# df = pd.DataFrame(values, columns=["File Name", "Maximum Eigen Ratio", "Variance of Eigen Ratio", "Normalized Area Under Eigen Ratio", "Area Under Eigen Ratio"])
df = pd.DataFrame(values)
df["Betti descriptor"] = []

df.to_csv(f"{pca_results_folder}/features.csv", index=False)
pca_features = df[["Max Eigen Ratio", "Variance of Eigen Ratio", "Normalized Area Under Eigen Ratio"]].values
tsne = TSNE(perplexity=4).fit_transform(pca_features)
# verify_tsne(pca_features, tsne)
distinct_labels = ["NON-STOCHASTIC", "STOCHASTIC"]
labels = [1] * len(files)
for i, gt in enumerate(df["Ground Truth"].values):
    if gt == "STOCHASTIC":
        labels[i] = 2

scatter_plot_2d(tsne,
                labels,
                pca_results_folder,
                "tsne",
                title="Tsne: Max Eigen Ratio, Variance and Area Under ER Curve",
                legends=distinct_labels,
                axis_labels=["Tsne Axis 1", "Tsne Axis 2"])

features = ["Max Eigen Ratio", "Variance of Eigen Ratio"]
scatter_plot_2d(df[features].values, labels, pca_results_folder, "max_eig_ratio_variance",
                title="Max Eigen Ratio and  Variance",
                legends= distinct_labels,
                axis_labels=features,
                log_axis=True)

features = ["Max Eigen Ratio", "Normalized Area Under Eigen Ratio"]
scatter_plot_2d(df[features].values, labels, pca_results_folder, "max_eig_ratio_area",
                title="Max Eigen Ratio and  Area",
                legends= distinct_labels,
                axis_labels=features,
                log_axis=True)

features = ["Variance of Eigen Ratio", "Normalized Area Under Eigen Ratio"]
scatter_plot_2d(df[features].values, labels, pca_results_folder, "variance_area",
                title="Variance and  Area",
                legends= distinct_labels,
                axis_labels=features,
                log_axis=True)

