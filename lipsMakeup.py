import cv2
import dlib
import numpy as np
hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame = cv2.imread("input.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = hog_face_detector(gray)

for face in faces:

    face_landmarks = dlib_facelandmark(gray, face)
    points = []
    for n in range(48, 60):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        points.append([x,y])
    x = face_landmarks.part(48).x
    y = face_landmarks.part(48).y
    points.append([x,y])
    for n in range(60, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        points.append([x,y])
    points = np.array(points)
    cv2.fillPoly(frame, [points], (0, 50., 199.))
    
cv2.imwrite("output.jpg", frame)
