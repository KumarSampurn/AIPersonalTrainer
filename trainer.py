import cv2
import time
import poseEstimationModule as pem
import numpy as np


def findpoints(img,lmList,muscle=0):
    if(muscle == 0): #right biceps points are 11 13 15 
        x1,y1= lmList[11][1], lmList[11][2]
        x2,y2= lmList[13][1], lmList[13][2]
        x3,y3= lmList[15][1], lmList[15][2]

        img =drawFeatures(img,(x1,y1),(x2,y2),(x3,y3))
        # angle =findAngle((x1,y1),(x2,y2),(x3,y3))
        
        
    return img


def drawFeatures(img,p1,p2,p3):
    x1,y1= p1[0],p1[1]
    x2,y2= p2[0],p2[1]
    x3,y3= p3[0],p3[1]
    
    
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.line(img,(x2,y2),(x3,y3),(0,255,0),2)
    
    cv2.circle(img,(x1,y1),3,(255,0,0),3,cv2.FILLED)
    cv2.circle(img,(x2,y2),3,(255,0,0),3,cv2.FILLED)
    cv2.circle(img,(x3,y3),3,(255,0,0),3,cv2.FILLED)

    return img



def putAngle(img,lmList,muscle=0):
        img=findpoints(img,lmList,muscle)
        
        
        

        return img






cap = cv2.VideoCapture(0)
# img=cv2.imread("AIPersonalTrainer/assets/2.jpg")
detecotor=pem.poseDetector(detectionCon=0.95,trackCon=0.95)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    # img=cv2.resize(img,(590,800))
    img = cv2.flip(img, 1)

    img=detecotor.findPose(img,draw=False)
    lmList=detecotor.findPosition(img,draw=False) 
    if(len(lmList)!=0):
        img= putAngle(img,lmList,0)       
                
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str((int)(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
