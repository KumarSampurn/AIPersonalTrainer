import cv2
import time
import poseEstimationModule as pem
import numpy as np

def checkAngle(img,count,goingUp,muscle=0):
        img,angle=detecotor.findAngleBetweenPoints(img,muscle)
        
        if(angle < 70 and goingUp==1):
            goingUp=0
            count=count+1
        if(angle > 120 and goingUp==0):
            goingUp=1  
       
        
        return img,count,goingUp

cap = cv2.VideoCapture("AIPersonalTrainer/assets/1.mp4")
detecotor=pem.poseDetector(detectionCon=.65,trackCon=.65)
pTime = 0
cTime = 0
goingUp=1
count=0
while True:
    success, img = cap.read()
    # img=cv2.resize(img,(590,800))
    img = cv2.flip(img, 1)

    img=detecotor.findPose(img,draw=False)
    lmList=detecotor.findPosition(img,draw=False) 
    if(len(lmList)!=0):
        img,count,goingUp= checkAngle(img,count,goingUp,0)       
    print("count :",count)  
    print("goingUp: ",goingUp)  
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str((int)(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
