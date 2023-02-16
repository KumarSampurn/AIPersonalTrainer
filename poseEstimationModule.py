import cv2
import mediapipe as mp
import time
import numpy as np



class poseDetector():
    
    def __init__(self,mode=False,modComplex=1,smooth_land=True,enable_Seg=False,smooth_seg=True,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.smooth_land=smooth_land
        self.enable_Seg=enable_Seg
        self.smooth_seg=smooth_seg
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.smooth_land,self.enable_Seg,self.smooth_seg, self.detectionCon,self.trackCon)
        
        
    def findPose(self,img,draw=True):
        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img 

    def findPosition(self, img, draw=True):
        
        self.lmlist=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy= (int)(lm.x * w) ,(int)(lm.y*h)
                self.lmlist.append((id,cx,cy))
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
                    
        return self.lmlist
    
    
    def findAngle(self,p1,p2, p3):
        
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
    
    
    def drawFeatures(self,img,p1,p2,p3):
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
    
    
    
           
    def findAngleBetweenPoints(self, img, muscle=0):
        if(muscle == 0): #right biceps points are 11 13 15 
            x1,y1= self.lmlist[11][1], self.lmlist[11][2]
            x2,y2= self.lmlist[13][1], self.lmlist[13][2]
            x3,y3= self.lmlist[15][1], self.lmlist[15][2]
            
        
            img =self.drawFeatures(img,(x1,y1),(x2,y2),(x3,y3))
            angle =int(self.findAngle((x1,y1),(x2,y2),(x3,y3)))
            cv2.putText(img,str(angle),(x2-50,y2-50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
                    
        return img, angle
        
 
def main():
    cap=cv2.VideoCapture("AIPersonalTrainer/assets/1.mp4")
    pTime=0
    detector=poseDetector()
    while True:
        success,img = cap.read()
        img=cv2.flip(img,1)
        
        img=detector.findPose(img,draw=True)
        lmlist=detector.findPosition(img,draw=True)
        if(len(lmlist)!=0):
            # print(lmlist[0])
            # cv2.circle(img,(lmlist[0][1],lmlist[0][2]),1,(255,0,0),cv2.FILLED)
            img,angle=detector.findAngleBetweenPoints(img,0)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
    
    
        cv2.putText(img,str((int)(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)
    
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    
if __name__ == "__main__":
    main()
    
    