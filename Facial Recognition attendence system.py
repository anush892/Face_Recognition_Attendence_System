# Importing all the reqiured Libraries 
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Adding a Video Capturing Device
video_capture = cv2.VideoCapture(0)

# Loading and Encoding all the Images
jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
jobs_encoding = face_recognition.face_encoding(jobs_image)[0]

elon_image = face_recognition.load_image_file("photos/elon.jpg") 
elon_encoding = face_recognition.face_encoding(elon_image)[0]

mona_image = face_recognition.load_image_file("photos/mona.jpg") 
mona_encoding = face_recognition.face_encoding(mona_image)[0]

einstein_image = face_recognition.load_image_file("photos/einstein.jpeg") 
einstein_encoding = face_recognition.face_encoding(einstein_image)[0]

known_face_encoding = [
jobs_encoding,
elon_encoding,
mona_encoding,
einstein_encoding
]

known_faces_names = [
"Steve Jobs",
"Elon Musk",
"Mona Lisa",
"Albert Einstein"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

# Getting Date & Time Information
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Creating CSV file with current Date & Time
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

# Comparing Faces Input from Database to find Similar Faces. 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_loactions(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_recognition(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
            
    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()