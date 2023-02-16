import cv2
import time
import poseEstimationModule as pem
import numpy as np

def checkAngle(img,count,goingUp,muscle=0):
        img,angle=detecotor.findAngleBetweenPoints(img,muscle)
        
        if(angle < 45 and goingUp==1):
            goingUp=0
            count=count+1
        if(angle > 145 and goingUp==0):
            goingUp=1  
       
        y=np.interp(angle,[45,120],[200,0])
       
        cv2.rectangle(img,(550,150),(600,350),(0, 255, 255),1)
        cv2.putText(img, str((int)(0.5*y))+"%", (550, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
        cv2.rectangle(img,(550,350-(int)(y)),(600,350),(0, 255, 255),-1)
        
        
        return img,count,goingUp

cap = cv2.VideoCapture(0)
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
        cv2.putText(img, "REPS : "+str((int)(count)), (350, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)  
          
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS :"+str((int)(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
