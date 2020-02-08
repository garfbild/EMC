import cv2
import numpy as np
from keras.datasets import mnist
import random
from matplotlib import pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
c = random.randint(0,10000)
im = x_train[c]
#im = cv2.imread("crocs.jpg",0)
#cv2.imshow('image',im)
print(c)
a = np.zeros(shape=(im.shape))
myfilter = np.array([[-1,-1,-1],[1,1,1],[0,0,0]])
rightfilter = np.array([[0,1,-1],[0,1,-1],[0,1,-1]])

for x in range(im.shape[1]-2):
    for y in range(im.shape[0]-2):
        a[y][x] = np.sum(np.multiply(np.array([[im[y][x],im[y][x+1],im[y][x+2]],[im[y+1][x],im[y+1][x+1],im[y+1][x+2]],[im[y+2][x],im[y+2][x+1],im[y+2][x+2]]]),rightfilter))

b = np.zeros(shape=(im.shape))
for x in range(im.shape[1]-2):
    for y in range(im.shape[0]-2):
        b[y][x] = np.sum(np.multiply(np.array([[im[y][x],im[y][x+1],im[y][x+2]],[im[y+1][x],im[y+1][x+1],im[y+1][x+2]],[im[y+2][x],im[y+2][x+1],im[y+2][x+2]]]),myfilter))

f = plt.figure(1)
plt.imshow(myfilter, cmap='gray')
f.show()

g = plt.figure(2)
plt.imshow(rightfilter, cmap='gray')
g.show()

