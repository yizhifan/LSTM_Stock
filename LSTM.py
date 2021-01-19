import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


def lstm_building(x_train, y_train, model_name=''):
    model = tf.keras.Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    # Adding the output layer
    model.add(tf.keras.layers.Dense(units=1))
    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    model.save(model_name + '.h5')


def validation(code, model, data, x_valid, y_valid):
    predicted_states = model.predict(x_valid)
    sc = MinMaxScaler(feature_range=(0, 1))
    valid = data.iloc[:, 1]
    valid = np.array(valid)
    valid = valid.reshape(-1, 1)
    sc.fit_transform(valid)
    predicted_states = sc.inverse_transform(predicted_states)
    real_states = sc.inverse_transform(y_valid.reshape(-1, 1))
    np.savetxt(code + 'predicted.csv', predicted_states, delimiter=',')
    np.savetxt(code + 'validation.csv', real_states, delimiter=',')

    plt.plot(real_states[:, 0], color='red', label='Real Price')
    plt.plot(predicted_states[:, 0], color='blue', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    plt.plot(real_states[-50: -1], color='red', label='Real Yaw')
    plt.plot(predicted_states[-51: -1], color='blue', label='Predicted Yaw')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
