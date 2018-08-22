import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
righteye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_found = False

    for (x,y,w,h) in faces:
        if w>0:
            face_found=True
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,'Face Detected',(0,13), font, 1, (255,0,0)) #---write the text

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        l_eye_found = False
        r_eye_found = False

        eyes = righteye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            if ew > 0:
                r_eye_found = True
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img,'Right Eye Detected',(50,130), font, 1, (255,0,0))
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            l_eyes = lefteye_cascade.detectMultiScale(roi_gray)
            for (lx,ly,lw,lh) in l_eyes:
                if lw >0 :
                    l_eye_found = True
                font = cv2.FONT_HERSHEY_PLAIN
                if l_eye_found == True :
                    cv2.putText(img,'Left Eye Detected',(90,90), font, 1, (255,0,0))
                    cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)
                else :
                    cv2.putText(img,'Blink',(90,90), font, 1, (255,0,0))
                    cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()