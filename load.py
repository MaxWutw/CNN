import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from urllib.request import urlretrieve
from PIL import Image

def pre_pic(picName):
    # 先打開傳入的原始圖片
    img = Image.open(picName)
    # 使用消除鋸齒的方法resize圖片
    reIm = img.resize((28,28),Image.ANTIALIAS)
    # 變成灰度圖，轉換成矩陣
    im_arr = np.array(reIm.convert("L"))
    return im_arr
   
img = pre_pic('picture.png')
x = img_to_array(img)

print('shape : ',x.shape)
plt.axis('off')
plt.imshow(x/255,cmap = 'Greys')
plt.show()
x = x.reshape(1, 28, 28, 1)

model = load_model('CNNmodel.h5')
result = model.predict_classes(x)
print('result is :' , result)
