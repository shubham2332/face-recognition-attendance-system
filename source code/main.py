import face_recognition
import numpy as np
import csv
import cv2
from datetime import datetime

video_capture = cv2.VideoCapture(0)

bhaiya_image = face_recognition.load_image_file("faces/bhaiya.jpg")
bhaiya_encoding = face_recognition.face_encodings(bhaiya_image)[0]

khushi_image = face_recognition.load_image_file("faces/khushi.jpg")
khushi_encoding = face_recognition.face_encodings(khushi_image)[0]

MSD_image = face_recognition.load_image_file("faces/MSD.png")
MSD_encoding = face_recognition.face_encodings(MSD_image)[0]

Kevin_image = face_recognition.load_image_file("faces/Kevin.jpeg")
Kevin_encoding = face_recognition.face_encodings(Kevin_image)[0]

sachintendulkar_image = face_recognition.load_image_file("faces/sachin tendulkar.jpg")
sachintendulkar_encoding = face_recognition.face_encodings(sachintendulkar_image)[0]

Shubham_image = face_recognition.load_image_file("faces/Shubham.jpg")
Shubham_encoding = face_recognition.face_encodings(Shubham_image)[0]

known_face_encodings = [bhaiya_encoding, khushi_encoding , MSD_encoding , Kevin_encoding , sachintendulkar_encoding, Shubham_encoding]
known_face_names = ["Bhaiya", "Khushi", "MSD", "Kevin", "sachintendulkar", "shubham"]

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "a", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontcolor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontcolor, thickness,
                        lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M:%S")  # Corrected format
                lnwriter.writerow([name, current_time])
                f.flush()  # Flush buffer to ensure data is written to the file immediately

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()