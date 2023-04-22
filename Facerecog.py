import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
Abhi_image = face_recognition.load_image_file("faces/Abhi.jpg")
Abhi_encoding = face_recognition.face_encodings(Abhi_image)[0]
sanu_image = face_recognition.load_image_file("faces/sanu.jpg")
sanu_encoding = face_recognition.face_encodings(sanu_image)[0]

known_face_encodings = [Abhi_encoding, sanu_encoding]
# use the variable name 'known_face_encodings' consistently throughout the code
known_face_names = ["Abhi", "sanu"]

students = known_face_names.copy()

face_locations = [] # use the variable name 'face_locations' consistently throughout the code
face_encodings = []

now = datetime.now()
current_date = datetime.strftime(now, "%Y-%m-%d") # fix the format string to convert 'now' to a string
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read() # fix the variable name and add a missing 'ret' variable
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) # fix the color conversion function name

    face_locations = face_recognition.face_locations(rgb_small_frame) # fix the variable name
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) # fix the variable name

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) # fix the function call to use 'argmin' instead of 'argmax'
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            lnwriter.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")]) # write the name and current time to the CSV file
            cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

f.close()
video_capture.release()
cv2.destroyAllWindows()
