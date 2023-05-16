import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp

import random

class FaceMesh2D:
   
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.contours = mp.solutions.face_mesh.FACEMESH_LIPS

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
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

        self.ax.clear()

        if not results.multi_face_landmarks:
            return
        
        face_landmarks = results.multi_face_landmarks[0].landmark

        x = [landmark.x for landmark in face_landmarks]
        y = [landmark.y for landmark in face_landmarks]
        z = [landmark.z for landmark in face_landmarks]
        
        a,b,c,d = FaceMesh2D.transform([x,y,z])
        #x,y,z = a,b,c
        
        w,h = image.shape[:2]
        x,y = np.transpose([mp.solutions.drawing_utils._normalized_to_pixel_coordinates(i,j,h,w) for i,j in zip(x,y)])

        xc = (x[291] + x[61])//2
        yc = (y[17] + y[0])//2
        ry = int((y[17] - y[0])*0.75)
        rx = int((x[291] - x[61])*0.75)
        ry = max(ry,9*rx//16)
        rx = max(rx,16*ry//9)

        image = image[yc-ry:yc+ry,xc-rx:xc+rx]
        x = [i-xc+rx for i in x]
        y = [j-yc+ry for j in y]

        self.ax.imshow(image)
        #self.ax.scatter(x, y, s=10)
        [self.ax.plot([x[i],x[j]],[y[i],y[j]],"black") for i,j in self.contours]

        #self.ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
        #id = [str(i) for i in range(len(x))]
        #[self.ax.text(x[i], y[i], z[i], i, fontsize=8) for i in range(len(x))

    def contour_cvt(self, points, contour):
        starters = [c[0]for c in contour]
        enders = [c[1]for c in contour]
        all_indices = []
        indices = []
        while(len(starters) > 0):
            try :
                try :
                    index = starters.index(indices[-1])
                    starters.pop(index)
                    indices.append(enders.pop(index))
                except:
                    index = enders.index(indices[-1])
                    enders.pop(index)
                    indices.append(starters.pop(index))
            except :
                all_indices.append(indices)
                indices = [starters.pop(0),enders.pop(0)]
        all_indices.append(indices)
        return [[[points[0][i],points[1][i]] for i in ind] for ind in all_indices[1:]]




    
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
    FaceMesh2D().start()



