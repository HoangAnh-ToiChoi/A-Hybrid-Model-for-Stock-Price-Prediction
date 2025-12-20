import tensorflow as tf
from tensorflow.keras.models import Sequential # mo hinh tuyen tinh, cac lop xep chong len nhau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, LSTM, Input # cac lop bulding block


def build_cnn_lstm_model(time_step = 60, features = 1, learning_rate = 0.001):
   model = Sequential()

   model.add(Input(shape=(time_step, features)))
   model.add(Conv1D(filters = 64, kernel_size = 2, activation = 'relu'))
   model.add(MaxPooling1D(pool_size = 2))

   model.add(LSTM(64, return_sequences = True))
   model.add(Dropout(0.2))

   model.add(LSTM(32, return_sequences = False))
   model.add(Dropout(0.2))

   model.add(Dense(1))

   optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
   model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer = optimizer)

   return model