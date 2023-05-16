import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp

import random

class FaceMesh3D:
   
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.contours = mp.solutions.face_mesh.FACEMESH_CONTOURS

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=270, azim=180)
        self.interval = 33.33 # 30 fps
        self.anim = FuncAnimation(self.fig, self.update, interval=self.interval)
        self.fig.canvas.mpl_connect('close_event', self.on_close)


    def start(self):
        plt.show()

    def on_close(self,event):
        self.face_mesh.close()
        self.cap.release()

        
    def update(self, frame):
        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        x,y,z = [],[],[]

        #x,y,z = [[random.randint(1, 9)/10 for i in range(4)]for i in range(3)]

        self.ax.clear()

        if not results.multi_face_landmarks:
            return
        
        face_landmarks = results.multi_face_landmarks[0].landmark

        x = [landmark.x for landmark in face_landmarks]
        y = [landmark.y for landmark in face_landmarks]
        z = [landmark.z for landmark in face_landmarks]
        
        #newPoints = FaceMesh3D.baseConvertion([x,y,z])
        a,b,c,d = FaceMesh3D.transform([x,y,z])
        x,y,z = a,b,c

        self.ax.scatter(x, y, z, s=1)
        [self.ax.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],"black") for i,j in self.contours]

        #self.ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
        #id = [str(i) for i in range(len(x))]
        #[self.ax.text(x[i], y[i], z[i], i, fontsize=8) for i in range(len(x))
    
    def transform(points):
        ids = ([151,9,8],[200,199,175],[33,133],[362,263])
        p1,p2,p3,p4 = [[ x for [x] in np.average(np.array([[points[i][id] for id in ids[j]] for i in range(len(points))]), axis=1, keepdims=True)] for j in range(len(ids))]

        vx = np.array(p2) - np.array(p1)
        vy = np.array(p4) - np.array(p3)
        vz = np.cross(vx,vy)
        vy = np.cross(vz,vx)

        vx = vx / np.linalg.norm(vx) / 10  
        vy = vy / np.linalg.norm(vy) / 10  
        vz = vz / np.linalg.norm(vz) / 10  

        centre = [p[9] for p in points]
        matpass = np.linalg.inv(np.vstack((np.vstack((vx, vy, vz,centre)).T,[0,0,0,1])))
        tPoints = np.transpose(np.vstack((points,[0]*len(points[0]))))
        return np.transpose([np.dot(matpass,p) for p in tPoints])


if __name__ == "__main__" :
    FaceMesh3D().start()



