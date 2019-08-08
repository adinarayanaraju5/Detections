'''Face detection along with smile detection '''

import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarCascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
cap.set(3,900) # set Width
cap.set(4,800) # set Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    userFaces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x,y,w,h) in userFaces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        userSmile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )

        for (xx, yy, ww, hh) in userSmile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

        cv2.imshow('video', img)

    k = cv2.waitKey(20) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
