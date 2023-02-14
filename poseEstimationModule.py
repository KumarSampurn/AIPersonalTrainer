import cv2
import mediapipe as mp
import time




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
        
        lmlist=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy= (int)(lm.x * w) ,(int)(lm.y*h)
                lmlist.append((id,cx,cy))
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,0),cv2.FILLED)
                    
        return lmlist
            
            
    
    
    
 
def main():
    cap=cv2.VideoCapture("poseEstimation/videos/1.mp4")
    pTime=0
    detector=poseDetector()
    while True:
        success,img = cap.read()
        img=cv2.flip(img,1)
        
        img=detector.findPose(img,draw=False)
        lmlist=detector.findPosition(img,draw=False)
        if(len(lmlist)!=0):
            # print(lmlist[0])
            cv2.circle(img,(lmlist[0][1],lmlist[0][2]),1,(255,0,0),cv2.FILLED)
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
    
    