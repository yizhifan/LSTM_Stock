import os
import tensorflow as tf
from get_data import get_data
from Data_Processing import data_norm, data_norm_predict
from LSTM import lstm_building, validation


if __name__ == '__main__':

    code = '002203'

    dataset_total = get_data(code)
    time_step = 50
    index = len(dataset_total)
    index = int(0.9 * index)
    train = dataset_total[0:index + time_step]
    valid = dataset_total[index - time_step:]
    if os.path.exists(code+'.h5'):
        model = tf.keras.models.load_model(code+'.h5')
        x_valid, y_valid = data_norm_predict(valid, time_step)
        validation(code, model, valid, x_valid, y_valid)
    else:
        X_train, y_train = data_norm(train, time_step)
        lstm_building(X_train, y_train, code)
        model = tf.keras.models.load_model(code + '.h5')
        x_valid, y_valid = data_norm_predict(valid, time_step)
        validation(code, model, valid, x_valid, y_valid)
