import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1) / 255
x_test = x_test.reshape(10000, 28, 28, 1) / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=(28,28,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(60, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

model.compile(loss = 'mse', optimizer = SGD(lr=0.087), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 128, epochs = 12)
result = model.predict_classes(x_test)
# result = np.argmax(model.predict(x_test), axis = -1)
loss, acc = model.evaluate(x_test, y_test)
print(f'accuracy: {acc*100:.2f}%')

def my_predict(n):
    print('CNN predict: ', result[n])
    X = x_test[n].reshape(28,28)
    plt.imshow(X, cmap='Greys')
n = 893
my_predict(n)
plt.show()
model.save('CNNmodel.h5')
