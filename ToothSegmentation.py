import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import mediapipe as mp
import keyboard

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
        self.ax1 = self.fig.add_subplot(2,2,1)
        self.ax2 = self.fig.add_subplot(2,2,2)
        self.ax3 = self.fig.add_subplot(2,2,3)
        self.ax4 = self.fig.add_subplot(2,2,4)
        self.interval = 33.33 # 30 fps
        self.anim = FuncAnimation(self.fig, self.update, interval=self.interval)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.treshcoeff = 0.3


    def start(self):
        plt.show()

    def on_close(self,event):
        self.face_mesh.close()
        self.cap.release()

        
    def update(self, frame):

        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        #image = cv2.imread("visage.jpg")
        image = cv2.flip(image, 1)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        x,y,z = [],[],[]

        #x,y,z = [[random.randint(1, 9)/10 for i in range(4)]for i in range(3)]

        if not results.multi_face_landmarks:
            return
        
        face_landmarks = results.multi_face_landmarks[0].landmark

        x = [landmark.x for landmark in face_landmarks]
        y = [landmark.y for landmark in face_landmarks]
        
        w,h = image.shape[:2]
        x,y = np.transpose([mp.solutions.drawing_utils._normalized_to_pixel_coordinates(i,j,h,w) for i,j in zip(x,y)])

        contour = self.contour_cvt((x,y),self.contours)
        mask = np.zeros((w, h, 1), dtype=np.uint8)
        cv2.fillPoly(mask,[np.int32(contour[1])],(255,255,255))
        image = cv2.bitwise_and(image, image, mask=mask)

        xc = (x[291] + x[61])//2
        yc = (y[17] + y[0])//2
        ry = int((y[17] - y[0])*0.5)
        rx = int((x[291] - x[61])*0.5)
        #ry = max(ry,9*rx//16)
        #rx = max(rx,16*ry//9)

        image = image[yc-ry:yc+ry,xc-rx:xc+rx]
        x = [i-xc+rx for i in x]
        y = [j-yc+ry for j in y]

        self.ax1.clear()
        self.ax1.imshow(image)

        traitedImage = self.toothSegmentation(image)
        self.ax2.clear()
        [self.ax2.plot([x[i],x[j]],[y[i],y[j]],"red") for i,j in self.contours]
        self.ax2.imshow(traitedImage)
        #self.ax.scatter(x, y, s=10)

    def toothSegmentation(self,image):
        b,grayImage,r = cv2.split(image)
        #grayImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        
        treshVal = self.findTresholdValue(grayImage)
        treshImage = cv2.threshold(grayImage,treshVal,255,cv2.THRESH_BINARY)[1]
        #treshImage = cv2.adaptiveThreshold(grayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,27,2)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=treshImage, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                            
        # draw contours on the original image
        image_copy = image.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        self.ax3.clear()
        self.ax3.imshow(image_copy,"gray")

        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(treshImage,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)

        rgbImage = cv2.cvtColor(dilation,cv2.COLOR_GRAY2RGB)
        return rgbImage
    
    def findTresholdValue(self,image):

        if keyboard.is_pressed("z"):
            self.treshcoeff += 0.01
        if keyboard.is_pressed("a"):
            self.treshcoeff += -0.01

        print(self.treshcoeff)

        self.ax4.clear()
        arr = []
        for line in image:
            arr += [line[i] for i in range(len(line))]
        sorted_arr = np.sort(arr)[::-1].tolist()
        if 0 in sorted_arr:
            sorted_arr = sorted_arr[:sorted_arr.index(0)]
        self.ax4.plot(sorted_arr)
        k = int(len(arr) * self.treshcoeff)
        top_k_values = sorted_arr[:k]
        avg_top_k_values = np.mean(top_k_values)
        self.ax4.plot([0,len(sorted_arr)],[avg_top_k_values,avg_top_k_values],c="red")
        return avg_top_k_values
    
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


if __name__ == "__main__" :
    FaceMesh2D().start()



