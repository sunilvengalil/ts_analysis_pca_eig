import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from utils import compute_svd, process_time_series_all, plot_eigen_ratio
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
file_name = "time_series"
M = 30
tau = 20
n = 5 # number of eigen vectors to visualize

tsne_visualization_results = f"visualize_{M}_tau_{tau}_top_{n}"

if not os.path.isdir(tsne_visualization_results):
    os.mkdir(tsne_visualization_results)

if os.path.isfile(file_name):
    files = [file_name]
else:
    files = os.listdir(file_name)

num_files = len(files)

data_dict = {}
max_value_dict = {}
for file in files:
    with open(file_name + "/" + file, "r") as fp:
        text = fp.readlines()
    data_dict[file] = np.asarray([float(d) for d in text])
    max_value = len(data_dict[file])
    max_value_dict[file] = max_value
    print(f"Number of samples in {file} {max_value}")

u_eig_vectors = np.zeros( (n * num_files, M))

for i, file_name in enumerate(data_dict.keys()):
    svd = compute_svd(data_dict[file_name], M, tau)
    u_eig_vectors[i * n: (i + 1) * n ] = svd[0][:n]

kmeans = KMeans(2).fit(u_eig_vectors)
print(kmeans.labels_.shape)
for i in range(num_files):
    print(files[i], kmeans.labels_[i * n : i * n + n])


#tsne()
# cluster()
# 1.  Each sample should have different weightage based on eigenvalue
# 2. instead of concatenations. - 2d
# 3. any other way of utilizing u u_eig_vectors
# 4. tsne_visualization_results
# 5. Use the features obtained from psyndy for binary classifacation
# 6. Use these eighen vectors as custom dictionary element


"""
1. area under the eigenratio cuve
2. maximum
3. Varaince of iegen ratio
4. shuffle the time sereis and run the same experiments
"""