import cv2
import time
import poseEstimationModule as pem
import numpy as np



def findAngle(p1,p2,p3):
    x1,y1= p1[0],p1[1]
    x2,y2= p2[0],p2[1]
    x3,y3= p3[0],p3[1]
    
    
    vector_a=[x1-x2,y1-y2]
    vector_b=[x3-x2,y3-y2]
    
    dot_product=np.dot(vector_a,vector_b)
    
    magnitude_a=np.linalg.norm(vector_a)
    magnitude_b=np.linalg.norm(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    
    theta = np.arccos(cos_theta)
    
    return(np.degrees(theta))
    



def drawFeatures(img,p1,p2,p3):
    x1,y1= p1[0],p1[1]
    x2,y2= p2[0],p2[1]
    x3,y3= p3[0],p3[1]
    
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.line(img,(x2,y2),(x3,y3),(0,255,0),2)
    
    cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
    cv2.circle(img,(x1,y1),15,(255,0,0),2)
    cv2.circle(img,(x2,y2),10,(255,0,0),cv2.FILLED)
    cv2.circle(img,(x2,y2),15,(255,0,0),2)
    cv2.circle(img,(x3,y3),10,(255,0,0),cv2.FILLED)
    cv2.circle(img,(x3,y3),15,(255,0,0),2)

    return img




def findDrawPoints(img,lmList,muscle=0):
    if(muscle == 0): #right biceps points are 11 13 15 
        x1,y1= lmList[11][1], lmList[11][2]
        x2,y2= lmList[13][1], lmList[13][2]
        x3,y3= lmList[15][1], lmList[15][2]

        img =drawFeatures(img,(x1,y1),(x2,y2),(x3,y3))
        angle =int(findAngle((x1,y1),(x2,y2),(x3,y3)))
        cv2.putText(img,str(angle),(x2-50,y2-50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                
    return img, angle


def checkAngle(img,lmList,count,goingUp,muscle=0):
        img,angle=findDrawPoints(img,lmList,muscle)
        
        if(angle < 70 and goingUp==1):
            goingUp=0
            count=count+1
        if(angle > 120 and goingUp==0):
            goingUp=1  
       
        
        return img,count,goingUp



cap = cv2.VideoCapture("AIPersonalTrainer/assets/1.mp4")
detecotor=pem.poseDetector(detectionCon=.5,trackCon=.5)
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
        img,count,goingUp= checkAngle(img,lmList,count,goingUp,0)       
    print("count :",count)  
    print("goingUp: ",goingUp)  
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str((int)(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
