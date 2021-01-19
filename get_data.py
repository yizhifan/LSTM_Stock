import tushare as tf


def get_data(code='', date='2010-01-01'):
    data = tf.get_k_data(code, date)
    data.to_csv(code + '.csv')
    train_data = data.iloc[:, 1:6]
    # print(train_data)
    return train_data
