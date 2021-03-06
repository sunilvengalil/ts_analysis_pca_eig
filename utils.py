import numpy as np
import sklearn
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import statistics

MIN_INTERVAL_SIZE = 100 # The smallest interval considered. If length of the interval is  <= 100, it is not splitted further
MAX_ITER = 50000 # Maximum number of iterations for splitting
MIN_EIGEN_VALUE = 10
MAX_NUM_SAMPLES_TO_TAKE = 1000

def split_series(series):
    """
    1. Subtract mean from the time series.
    1. Split the time series into two equal halves.
    2. Compute eigenvalue of each half, find the ratio of larger to smaller eigen value.
    3.
    """
    series = series - np.mean(series)
    if series.shape[0] % 2 == 0:
        # even
        series_1 = series[0:series.shape[0] // 2]
        series_2 = series[series_1.shape[0]:]
    else:
        series_1 = series[0:series.shape[0] // 2]
        series_2 = series[series_1.shape[0]:-1]

    # print("var1 converted to float", float(var1))
    zipped = []
    for i, j in zip(series_1, series_2):
        zipped.append([i, j])

    cov = np.cov(np.transpose(np.asarray(zipped)))
    e1, _ = np.linalg.eig(np.asarray(cov))

    eig_ratio = max(e1) / min(e1)
    return eig_ratio


def process_time_series_all(s, min_eig_value=MIN_EIGEN_VALUE):
    """
    :param s:
    :param min_eig_value:
    :return: A list with elements of the form [eigenvalue_ratio, interval]
    """
    interval = (0, len(s))
    results = []
    splits_to_process = []
    num_iter = 0
    num_splits = 0

    eig_ratio = split_series(s[interval[0]:interval[1]])
#    print("eig_ratio", eig_ratio)
    #     if eig_ratio > min_eig_value:
    #         results.append((eig_ratio, interval))

    if interval[1] - interval[0] > MIN_INTERVAL_SIZE:
        mid = (interval[1] + interval[0]) // 2
        if eig_ratio < min_eig_value:
#            print("Splitting further")
            splits_to_process.append([interval[0], mid])
            splits_to_process.append([mid, interval[1]])
            num_splits += 1

    while len(splits_to_process) > 0 and num_iter < MAX_ITER:
#        print("Length of split to proecess",len(splits_to_process))
        interval = splits_to_process.pop()
        eig_ratio = split_series(s[interval[0]:interval[1]])
        #        if eig_ratio > min_eig_value:
 #       print(eig_ratio, interval)
        results.append((eig_ratio, interval))
        if interval[1] - interval[0] > MIN_INTERVAL_SIZE:
            mid = (interval[1] + interval[0]) // 2
            if eig_ratio < min_eig_value:
                splits_to_process.append([interval[0], mid])
                splits_to_process.append([mid, interval[1]])
                num_splits += 1
        num_iter = num_iter + 1

    return results, num_splits, num_iter



def plot_eigen_ratio(data, eigenvalue_ratios, num_samples, fname,
                     title = None,
                     y_scale=None,
                     xticks=None,
                     x_scale=None):
    """
    :param data: 
    :param eigenvalue_ratios: A list of the form [eigenvalue_ratio, interval] where interval is tuple of the form (from_index, to_index)
    :param num_samples: 
    :param fname: 
    :param title: 
    :param y_scale: 
    :return: 
    """
    result_step = np.zeros(num_samples)
    for r in eigenvalue_ratios:
        result_step[r[1][0]: r[1][1]] = r[0]

    plt.figure(figsize=(20, 8))
    plt.rc('axes', labelsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)

   # plt.subplots(211)
    plt.plot(result_step)
    if xticks is not None:
        plt.xticks(xticks)
    if y_scale is not None:
        plt.ylim(y_scale)
    if x_scale is not None:
        plt.xlim(x_scale)
    #plt.xlabel("Sample Number")
    #plt.ylabel("Eigenvalue ratio")
    plt.title(title)

    # plt.subplots(212)
    # plt.plot(data)
    plt.savefig(fname, bbox_inches="tight" )


def plot_time_series(data, fname, title, y_scale=None, xticks=None, x_scale=None):
    """
    :param data:
    :param eigenvalue_ratios: A list of the form [eigenvalue_ratio, interval]
    :param num_samples:
    :param fname:
    :param title:
    :param y_scale:
    :return:
    """

    plt.figure(figsize=(10, 8))
    plt.rc('axes', labelsize=25)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    if y_scale is not None:
        plt.ylim(y_scale)
    if x_scale is not None:
        plt.xlim(x_scale)
    plt.plot(data)
    if xticks is not None:
        plt.xticks(xticks)
    plt.xlabel("Sample number")
    plt.ylabel("Photon count rate")
    plt.title(title)
    plt.savefig(fname, bbox_inches="tight")


def area_per_unit_length(results, max_value):
    result = 0
    for r in results:
        result += (r[1][1] - r[1][0]) * r[0]
    result /= max_value
    return result

def print_results(data_name, results, max_value, num_splits):
    min_eig_ratio = min([r[0] for r in results])
    max_eig_ratio = max([r[0] for r in results])
    var_eig_ratio = statistics.variance([r[0] for r in results])
    area = area_per_unit_length(results, max_value)
    print(f"***********{data_name}***********")
    print("Minimum Eigen Ratio", min_eig_ratio)
    print("Maximum Eigen Ratio", max_eig_ratio)
    print("Variance of  Eigen", var_eig_ratio)
    print("Area of eig_ratio", area)
    print("Number of splits", num_splits)


def compute_svd(data, M, tau, max_num_samples_to_take = MAX_NUM_SAMPLES_TO_TAKE):
    shifted_for_svd = create_image_data(M, data, max_num_samples_to_take, tau)
    svd = np.linalg.svd(shifted_for_svd)
    return svd


def create_image_data( data, M,  tau,max_num_samples_to_take= MAX_NUM_SAMPLES_TO_TAKE):
    num_samples_to_take = min(data.shape[0] - M * tau, max_num_samples_to_take)
    shifted_for_svd = np.zeros((M, num_samples_to_take))
    for i in range(M):
        shifted_for_svd[i] = data[i * tau:][0:num_samples_to_take]
    shifted_for_svd = np.asarray(shifted_for_svd)
    return shifted_for_svd


def compute_pca_on_embedded_vecors(data, M, tau, max_num_samples_to_take=MAX_NUM_SAMPLES_TO_TAKE):
    num_samples_to_take = min(data.shape[0] - M * tau, max_num_samples_to_take)

    shifted_for_svd = np.zeros((M, num_samples_to_take))
#    shifted_for_svd
    for i in range(M):
        shifted_for_svd[i] = data[i * tau:][0:num_samples_to_take]
    shifted_for_svd = np.asarray(shifted_for_svd)

    # print(shifted_for_svd.shape)
    # print(len(shifted_for_svd[0]))
#     eigs,_ = np.linalg.eig(np.matmul(shifted_for_svd, np.transpose(shifted_for_svd)))
    pca = PCA(M)
    pca.fit(np.transpose(shifted_for_svd))

    return  pca.singular_values_


def verify_tsne(data, tsne):
    data_distance = np.zeros((data.shape[0], data.shape[0]))
    tsne_distance = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            data_distance[i, j] = np.linalg.norm(data[i, :] - data[j, :])
            tsne_distance[i, j] = np.linalg.norm(tsne[i, :] - tsne[j, :])
    print("Data distance")
    print(data_distance)

    print("Tsne distance")
    print(tsne_distance)


def scatter_plot_2d(data: dict,
                    labels, out_folder, file_name, title=None, legends=None, axis_labels=None, log_axis = False):
    print("Entire data",data)
    cdict = {1: 'red', 2: 'blue'}
    mdict = {1: 'o',  2: 'x'}
    plt.figure()
    if log_axis:
        xy = np.log(data)
    else:
        xy = data
    print(labels)
    for g in np.unique(labels):
        ix = np.where(labels == g)
        print(g, xy)
        print(xy[ix, 0], xy[ix, 1], cdict[g], mdict[g], legends)
        plt.scatter(xy[ix, 0], xy[ix, 1], c=cdict[g], marker = mdict[g], label=legends[g - 1])

    plt.legend()
    plt.title(title)
    xlabel = axis_labels[0]
    ylabel = axis_labels[1]
    if log_axis:
        xlabel = f"Log({xlabel})"
        ylabel = f"Log({ylabel})"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{out_folder}/{file_name}.jpg", bbox_inches="tight")
    print(" Ploted scatter plot")