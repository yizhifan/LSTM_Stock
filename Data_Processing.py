import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


def data_norm(data, time_step=20, feature_range=0):
    sc = MinMaxScaler(feature_range=(feature_range, 1))
    training_set_scaled = sc.fit_transform(data)
    x_data = []
    y_data = []
    for i in range(time_step, len(data)):
        x_data.append(training_set_scaled[i - time_step:i])
        y_data.append(training_set_scaled[i, 1])
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def data_norm_predict(data, time_step, feature_range=0):
    sc = MinMaxScaler(feature_range=(feature_range, 1))
    valid_set_scaled = sc.fit_transform(data)
    x_valid = []
    y_valid = []
    for i in range(time_step, len(data) + 1):
        x_valid.append(valid_set_scaled[i - time_step:i])
    for i in range(time_step, len(data)):
        y_valid.append(valid_set_scaled[i, 1])
    x_valid, y_valid = np.array(x_valid), np.array(y_valid)
    return x_valid, y_valid
