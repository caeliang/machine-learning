import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
# filter warnings
warnings.filterwarnings('ignore')


x_l = np.load('C://Users//ranim//source//ml//sign language digits//X.npy')
Y_l = np.load('C://Users//ranim//source//ml//sign language digits//Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')