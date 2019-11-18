import numpy as np
import cv2

cap0 = cv2.VideoCapture('../dataset/4p-c0.avi')
cap1 = cv2.VideoCapture('../dataset/4p-c1.avi')
cap2 = cv2.VideoCapture('../dataset/4p-c2.avi')
cap3 = cv2.VideoCapture('../dataset/4p-c3.avi')

while(cap0.isOpened()):
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if frame0 is None:
    	break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("0.jpg", gray0)
    cv2.imwrite("1.jpg", gray1)
    cv2.imwrite("2.jpg", gray2)
    cv2.imwrite("3.jpg", gray3)


    cv2.imshow('frame',gray0)
    cv2.imshow('frame',gray1)
    cv2.imshow('frame',gray2)
    cv2.imshow('frame',gray3)

    break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cv2.destroyAllWindows()
