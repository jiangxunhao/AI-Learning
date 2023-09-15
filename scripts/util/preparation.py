import numpy as np


def load(data_arg):
    # get the path of training data file
    features_file_path = data_arg + "/instance-features.txt"
    performances_file_path = data_arg + "/performance-data.txt"

    # load the data
    features = np.loadtxt(features_file_path, delimiter=" ")
    performances = np.loadtxt(performances_file_path, delimiter=" ")
    return features, performances


def standardise(data, mean=None, std=None):
    mean = mean if mean is not None else np.mean(data, axis=0)
    std = std if std is not None else np.std(data, axis=0)
    np.savetxt('mean.txt', mean)
    np.savetxt('std.txt', std)
    return (data - mean) / np.where(std != 0, std, 1)


def normalise(data):
    col_max = data.max(axis=0)
    col_min = data.min(axis=0)
    return (data - col_min) / np.where((col_max-col_min) != 0, col_max-col_min, 1)


def constant_columns(data):
    return np.argwhere(data.max(axis=0) == data.min(axis=0))


def remove_constant(data, constant_cols):
    return np.delete(data, constant_cols, axis=1)


