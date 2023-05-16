# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FILES = ["visage.jpg"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    
    face_landmarks = results.multi_face_landmarks[0].landmark
    x = [landmark.x for landmark in face_landmarks]
    y = [landmark.y for landmark in face_landmarks]
    z = [landmark.z for landmark in face_landmarks]

    # Change the Size of Graph using
    # Figsize
    fig = plt.figure(figsize=(10, 10))
    
    # Generating a 3D sine wave
    ax = plt.axes(projection='3d')
    
    # To create a scatter graph
    ax.scatter(x, y, z)
    print(np.transpose(np.array([x, y, z])))
    [ax.plot([x[i],x[j]],[y[i],y[j]],[z[i],z[j]],"black") for i,j in mp_face_mesh.FACEMESH_CONTOURS]

    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], str(i))
    
    # turn off/on axis
    #plt.axis('off')
    
    # show the graph
    plt.show()

    