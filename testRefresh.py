import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp

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
        self.ax.view_init(elev=270, azim=270)
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
        image = cv2.cvtColor(image, cv2.BGR2RGB)
        results = self.face_mesh.process(image)

        x,y,z = [],[],[]

        self.ax.clear()

        if not results.multi_face_landmarks:
            return
        
        face_landmarks = results.multi_face_landmarks[0].landmark

        x = [landmark.x for landmark in face_landmarks]
        y = [landmark.y for landmark in face_landmarks]
        z = [landmark.z for landmark in face_landmarks]
        
        #newPoints = FaceMesh3D.baseConvertion([x,y,z])
        a,b,c = FaceMesh3D.transform([x,y,z])

        self.ax.scatter(x, y, z, s=1)
        [self.ax.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],"black") for i,j in self.contours]

        #self.ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
        #id = [str(i) for i in range(len(x))]
        #[self.ax.text(x[i], y[i], z[i], i, fontsize=8) for i in range(len(x))

    def baseConvertion(points):
        startBase = np.identity(3)
        endBase = FaceMesh3D.base(points)
        passa = FaceMesh3D.matPass(startBase,endBase)
        return [np.dot(passa,np.array([points[0][i],points[1][i],points[2][i]])) for i in range(len(points[0]))]


    def base(points):
        # xVect
        xIds= ((33,133),(362,263))
        xPoints = (FaceMesh3D.extract(points, xIds[0]),FaceMesh3D.extract(points, xIds[1]))
        xPointsMoy = (FaceMesh3D.moy(xPoints[0]),FaceMesh3D.moy(xPoints[1]))
        xVect = FaceMesh3D.sub(xPointsMoy[1],xPointsMoy[0])
        xVectNorm = FaceMesh3D.norm(xVect)
        xVect = FaceMesh3D.div(xVect,xVectNorm)

        # yVect
        yIds = ((8,9,151),(175,199,200))
        yPoints = (FaceMesh3D.extract(points, yIds[0]),FaceMesh3D.extract(points, yIds[1]))
        yPointsMoy = (FaceMesh3D.moy(yPoints[0]),FaceMesh3D.moy(yPoints[1]))
        yVect = FaceMesh3D.sub(yPointsMoy[1],yPointsMoy[0])
        yVectNorm = FaceMesh3D.norm(yVect)
        yVect = FaceMesh3D.div(yVect,yVectNorm)

        # zVect
        zVect = FaceMesh3D.vect3D(xVect,yVect)

        return (xVect,yVect,zVect)
    
    def transform(points):
        ids = (8,200,33,263)
        p1,p2,p3,p4 = [[points[i][id] for i in range(len(points))] for id in ids]
        vx= np.array(p2) - np.array(p1)
        vy = np.array(p4) - np.array(p2)
        a,b,c = [[p1[i],p2[i]] for i in range(len(p1))]
        d,e,f = [[p3[i],p4[i]] for i in range(len(p3))]
        plt.plot(a,b,c)
        plt.plot(d,e,f)
        vz = np.cross(vx,vy)
        plt.plot([p1[0],vz[0]+p1[0]],[p1[1],vz[1]+p1[1]],[p1[2],vz[2]+p1[2]],"blue")
        vy = np.cross(vz,vx)
        vx = vx / np.linalg.norm(vx) / 10  
        vy = vy / np.linalg.norm(vy) / 10  
        vz = vz / np.linalg.norm(vz) / 10  
        matpass = np.linalg.inv(np.vstack((vx, vy, vz)))
        tPoints = np.transpose(points)
        #plt.plot([p1[0],vx[0]+p1[0]],[p1[1],vx[1]+p1[1]],[p1[2],vx[2]+p1[2]],"red")
        #plt.plot([p1[0],vy[0]+p1[0]],[p1[1],vy[1]+p1[1]],[p1[2],vy[2]+p1[2]],"green")
        #plt.plot([p1[0],vz[0]+p1[0]],[p1[1],vz[1]+p1[1]],[p1[2],vz[2]+p1[2]],"blue")
        return np.transpose([np.dot(matpass,p) for p in tPoints])



        

    def matPass(matA,MatB):
        return np.dot(np.linalg.inv(matA),MatB)

    def extract(points, ids):
        return [[p[id] for id in ids] for p in points]
        
    def moy(points):
        return [sum(p)/len(p) for p in points]

    def sub(pointA,pointB):
        return [a-b for a,b in zip(pointA,pointB)]
    
    def div(vect,val):
        return [v/val for v in vect]
        
    def norm(vect):
        return np.sqrt(sum([v**2 for v in vect]))
    
    def vect3D(vectA,VectB):
        index = ((1,2),(2,0),(0,1))
        return [vectA[i]*VectB[j]-vectA[j]*VectB[i] for i,j in index]


if __name__ == "__main__" :
    FaceMesh3D().start()



