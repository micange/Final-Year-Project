import cv2
import numpy as np

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SimgPLEX, 1, 1, 0, 1, 1)

while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf&lt;50):
            if(Id==1):
                Id="Person 1"
            elif(Id==2):
                Id="Person 2"
        else:
            Id="Not in dataset"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(Id), (x,y+h),font, 255)
    cv2.imgshow('image',img) 
    if cv2.waitKey(10) &amp; 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

