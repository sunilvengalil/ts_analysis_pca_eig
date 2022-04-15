from collections import defaultdict
from matplotlib import pyplot as plt

from utils import compute_svd, process_time_series_all
import os
import numpy as np
tau_array = [20]
M_array = [2]
max_num_samples_to_take = 5000
MIN_EIGEN_VALUE = 10


# file_names = ["Delta", "Theta", "Beta", "Kai", "Lorenz", "Lordata", "Lambda", "Kappa", "Phi", "Gamma", "Alpha", "Mu", "Nu", "Rho"]
# file_names =["Mu", "Nu","Alpha", "Delta"]
# ground_truths = ["STOCHASTIC", "NON-ST", "NON-ST", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST", "STOCHASTIC", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST"]
# gt_dictionary = {file_name:ground_truth for file_name, ground_truth in zip(file_names, ground_truths) }


file_name = "time_series"
shuffle = False

time_series = "kappa"
if shuffle:
    svd_results_folder = f"svd_results_{time_series}_m_{M_array[0]}_{M_array[-1]}_tau_{tau_array[0]}_{tau_array[-1]}_shuffled_v2"
else:
    svd_results_folder = f"svd_results_{time_series}_m_{M_array[0]}_{M_array[-1]}_tau_{tau_array[0]}_{tau_array[-1]}_v2"

if not os.path.isdir(svd_results_folder):
    os.mkdir(svd_results_folder)


if os.path.isfile(file_name):
    files = [file_name]
else:
    files = os.listdir(file_name)


data_dict = {}
max_value_dict = {}
print(files)
for file in files:
    print(file)
    with open(file_name + "/" + file, "r") as fp:
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
            results, num_splits, num_iters = process_time_series_all(np.asarray(data_dict[file_name]),
                                                                     min_eig_value=MIN_EIGEN_VALUE)
            # plot_eigen_ratio(data_dict[file_name], results, max_value_dict[file_name],
            #                  f"{pca_results_folder}/{file_name}.jpg",
            #                  title=f"{file_name}: Eigen ratio greater than {MIN_EIGEN_VALUE}")
            svd = compute_svd(data_dict[file_name], M, tau)
            v_mat = svd[2]
            plt.figure()
            plt.scatter(v_mat[0], v_mat[1])
            plt.xlabel("E1", weight="bold")
            plt.ylabel("E2", weight="bold")
            plt.savefig(f"{svd_results_folder}/{file_name}_e1_vs_e2_m_{M}_tau_{tau}.jpg", bbox_inches="tight")

