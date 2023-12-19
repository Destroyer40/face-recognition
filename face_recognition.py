import face_recognition
import cv2
import numpy as np

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Load reference image and compute its face encoding
known_image = face_recognition.load_image_file("image.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Lists to store known face encodings and names
known_face_encodings = [known_face_encoding]
known_face_names = ["Your Name"]

process_frame = True

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert color space
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_frame:
        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            # Compare faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)
    
    process_frame = not process_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.rectangle(frame, (left, bottom - 30), (right + 45, bottom), (0, 0, 0), cv2.FILLED)
        
        # Add name label
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 3, bottom - 4), font, 1.0, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Press Q to Quit', frame)

    # Break the loop if 'q' or 'Q' is pressed
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
