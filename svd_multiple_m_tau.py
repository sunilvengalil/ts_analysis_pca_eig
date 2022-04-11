from collections import defaultdict
from matplotlib import pyplot as plt

from utils import compute_svd, process_time_series_all
import os
import numpy as np
M_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
tau_array = [80]
max_num_samples_to_take = 5000
MIN_EIGEN_VALUE = 10


#file_names = ["Delta", "Theta", "Beta", "Kai", "Lorenz", "Lordata", "Lambda", "Kappa", "Phi", "Gamma", "Alpha", "Mu", "Nu", "Rho"]
# file_names =["Mu", "Nu","Alpha", "Delta"]
#ground_truths = ["STOCHASTIC", "NON-ST", "NON-ST", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST", "STOCHASTIC", "STOCHASTIC", "NON-ST", "NON-ST", "NON-ST", "NON-ST"]
#gt_dictionary = {file_name:ground_truth for file_name, ground_truth in zip(file_names, ground_truths) }


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
print(files)
for file in files[1:]:
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
            plt.plot(v_mat[0], v_mat[1])
            plt.xlabel("E1")
            plt.ylabel("E2")
            plt.savefig(f"{svd_results_folder}/{file_name}_e1_vs_e2_m_{M}_tau_{tau}.jpg")




#
#
#         svd = compute_svd(M, tau, num_samples_to_take=max_num_samples_to_take, )
#         v_mat = svd[2]
#         fig = plt.figure()
#         plt.plot(v_mat[0][0:5000], v_mat[1][0:5000])
#         plt.xlabel("E1")
#         plt.ylabel("E2")
# #        plt.title( f"M={M}  tau={tau}")
#         plt.axis("off")
#         plt.savefig(f"Results/Lorenz/Change_M_Fix_Tau/Lorenz_e1_vs_e2_{M}_{tau}.jpg")
