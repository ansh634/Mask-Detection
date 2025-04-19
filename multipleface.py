import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_and_crop_faces(frame):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Return all detected faces and their locations
    return [(frame[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]

def detect_mask_in_frame(face_crop, model, Image_size=100):
    # Resize the face crop for model input
    resized_face = cv2.resize(face_crop, (Image_size, Image_size))
    resized_face = np.expand_dims(resized_face / 255.0, axis=0)

    # Predict mask status
    prediction = model.predict(resized_face)
    predicted_class = ['Masked', 'Not Masked'][np.argmax(prediction)]

    return predicted_class, prediction[0][np.argmax(prediction)]

def live_mask_detection(model, Image_size=100):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect multiple faces and their locations
        detected_faces = detect_and_crop_faces(frame)

        for face_crop, (x, y, w, h) in detected_faces:
            # Run mask detection on each cropped face
            predicted_class, confidence = detect_mask_in_frame(face_crop, model, Image_size)

            # Annotate the frame with mask detection result
            color = (0, 255, 0) if predicted_class == 'Masked' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Display the frame
        cv2.imshow('Live Mask Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model('model.h5')
    live_mask_detection(model, Image_size=100)
