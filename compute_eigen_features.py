import statistics
from collections import defaultdict

import numpy as np
import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from utils import compute_svd, process_time_series_all, plot_eigen_ratio, area_per_unit_length
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
M = 10
tau = 20

svd_results_folder = f"svd_results_m_{M}_tau_{tau}"
pca_results_folder = f"pca_results_m_{M}_tau_{tau}"

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
    data_dict[file] = np.asarray([float(d) for d in text])
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

    # insert into dictionary for creating dataframe
    values["File Name"].append(file_name)
    values["Max Eigen Ratio"].append(max_eig_ratio)
    values["Variance of Eigen Ratio"].append(var_eig_ratio)
    values["Normalized Area Under Eigen Ratio"].append(normalized_area)
    key = file_name.rsplit(".",  1)[0]
    print(key)
    if key.startswith("sac_ascf_"):
        key = key.replace("sac_ascf_", "")
    print(key)
    values["Ground Truth"].append(gt_dictionary[key.capitalize()])


# df = pd.DataFrame(values, columns=["File Name", "Maximum Eigen Ratio", "Variance of Eigen Ratio", "Normalized Area Under Eigen Ratio", "Area Under Eigen Ratio"])

df = pd.DataFrame(values)
df.to_csv(f"{pca_results_folder}/features.csv", index=False)
tsne = TSNE().fit_transform(df[["Max Eigen Ratio", "Variance of Eigen Ratio", "Normalized Area Under Eigen Ratio"]].values)
print(tsne.shape)

plt.scatter(tsne[:, 0], tsne[:, 1])
plt.title("Tsne: Max Eigen Ratio, Variance and Area Under ER Curve")
plt.xlabel("Tsne Axis 1")
plt.ylabel("Tsne Axis 2")
plt.savefig(f"{pca_results_folder}/tsne.jpg")
