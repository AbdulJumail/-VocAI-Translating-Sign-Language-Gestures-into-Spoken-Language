import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def recognize_emotion(frame, face_classifier, emotion_model, emotion_labels):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            prediction = emotion_model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            labels.append(label)
        else:
            labels.append('No Faces')

    return labels

def integrate_emotion_and_sign_language():
    cap = cv2.VideoCapture(0) 
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    with holistic:
        # Load the pre-trained models
        sign_model = load_model('action.h5')
        emotion_model = load_model('model.h5')
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        sequence = []  # Initialize an empty sequence
        recognized_sentence = ""  # Initialize an empty sentence
        sentence_delay = 5  # Set the sentence delay in seconds
        last_recognition_time = time.time()  # Initialize last recognition time

        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            # Draw Landmarks
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)

            sequence.insert(0, keypoints)  # Insert the new frame into the sequence
            sequence = sequence[:30]  # Keep only the last 30 frames

            # Recognize sign language
            if len(sequence) == 30:
                sequence_array = np.array(sequence)
                res = sign_model.predict(np.expand_dims(sequence_array, axis=0))
                recognized_action = actions[np.argmax(res[0])]

                cv2.putText(image, f"Recognized Action: {recognized_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if time.time() - last_recognition_time >= sentence_delay:
                    recognized_sentence += recognized_action + " "  # Add the recognized word to the sentence
                    last_recognition_time = time.time()

                # Recognize emotion
                emotion_labels = recognize_emotion(frame, face_classifier, emotion_model, ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
                for i, label in enumerate(emotion_labels):
                    cv2.putText(image, f"Emotion {i+1}: {label}", (10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Combine recognized sentence and emotion
                complete_sentence = recognized_sentence.strip() + f" [{', '.join(emotion_labels)}]"

                cv2.putText(image, f"Complete Sentence: {complete_sentence}", (10, 120 + len(emotion_labels)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return complete_sentence

if __name__ == "__main__":
    actions = ['hello', 'thanks', 'iloveyou', 'beautiful', 'happy', 'loud']
    recognized_sentence = integrate_emotion_and_sign_language()
    print("Recognized Sentence:", recognized_sentence)
