import cv2
import os
import numpy as np
from PIL import Image

recogniser = cv2.face.createLBPHFaceRecognizer()
haar= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def retImgAndLabel(path):
    
	#get the paths of all the images inside the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
	
    #create list for faces
    faceImg=[]
   
	#create a list for id
    Ids=[]
	
    #loop image paths
    for imagePath in imagePaths:
	
        #convert to grayscale
        pilImage=Image.open(imagePath).convert('L')
        #Convert pill to numpy
        imageNp=np.array(pilImage,'uint8')
        #get id of image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        #extract face from sample
        faces=haar.detectMultiScale(imageNp)
        #Append face and id in list if present
        for (x,y,w,h) in faces:
            faceImg.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceImg,Ids


faces,Ids = retImgAndLabel('dataSet')
recogniser.train(faces, np.array(Ids))
recogniser.save('trainner/trainner.yml')
cv2.destroyAllWindows()

