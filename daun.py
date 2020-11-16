import numpy as np

import cv2

import matplotlib.pyplot as plt

src = cv2.imread('/home/leechanhyuk/Desktop/bad.jpg')
src = cv2.resize(src,dsize=(230,730))
bgr = ('b','g','r')
plt.figure(1)
for i,col in enumerate(bgr):
    histogram = cv2.calcHist([src],[i],None,[256],[0,256])
    plt.plot(histogram,color = col)
    plt.xlim([0,256])
plt.show()

b,g,r = cv2.split(src)

high=0
low=255

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if r[i,j]>high:
            high=r[i,j]
        if r[i,j]<low:
            low=r[i,j]
r_high = high
r_low = low
high=0
low=255
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if b[i,j]>high:
            high=b[i,j]
        if b[i,j]<low:
            low=b[i,j]
b_high = high
b_low = low
high=0
low=255
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        if g[i,j]>high:
            high=g[i,j]
        if r[i,j]<low:
            low=g[i,j]
g_high = high
g_low = low

r_value = 255.0/(r_high - r_low)
g_value = 255.0/(g_high - g_low)
b_value = 255.0/(b_high - b_low)

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        r[i,j] = (r[i,j] - r_low) * r_value
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        g[i,j] = (g[i,j] - g_low) * g_value
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        b[i,j] = (b[i,j] - b_low) * b_value
final = cv2.merge((b,g,r))
cv2.imshow("src",src)
cv2.waitKey(1)
cv2.imshow("final",final)
cv2.waitKey(0)


plt.figure(2)
histogram = cv2.calcHist([r],[0],None,[256],[0,256])
plt.plot(histogram,color = 'r')
histogram = cv2.calcHist([g],[0],None,[256],[0,256])
plt.plot(histogram,color = 'g')
histogram = cv2.calcHist([b],[0],None,[256],[0,256])
plt.plot(histogram,color = 'b')
plt.xlim([0,256])
plt.show()





