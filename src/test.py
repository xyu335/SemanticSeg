#!/usr/bin/python3
import cv2
import numpy as np

filename = "./cam0.png"
output = "./cam0_test.png"

x1 = 20
x2 = 200
y1 = 20 
y2 = 200 
img = cv2.imread(filename)
cv2.rectangle(img, (x1,y1), (x2,y2), (50,50,50), thickness=2)
b,g,r = cv2.split(img)
print(np.shape(b))
sh = np.shape(img)
mask = np.zeros((sh[0], sh[1]), np.uint8)

for i in range(200):
    for j in range(200):
        mask[i][j] = 255
for i in range(200, 400):
    for j in range(200, 400):
        mask[i][j] = 125

# new_img = cv2.merge((b,g,r,mask))
# cv2.imwrite(output, new_img)

def test(): 
    a = 20
    b = 10
    return (None,b)

d, e = test()
c = test()
print(type(c))
print("{0},{1}".format(d,e))
