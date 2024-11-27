import cv2
import os
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
import json
#from tensorflow.keras import layers, models
from collections import deque
from PIL import Image, ImageDraw, ImageFont

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

# Load the model and landmarks from files
model_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'sign_language_model.keras')
label_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'filtered_landmarks_dataset.pkl')

model = tf.keras.models.load_model(model_path)
with open(label_path, "rb") as f:
    filtered_landmarks_dataset = pickle.load(f)

# Create a word index from the filtered landmarks dataset
label_encoder = tf.keras.preprocessing.text.Tokenizer()
label_encoder.fit_on_texts([data['word'] for data in filtered_landmarks_dataset])
word_index = label_encoder.index_word

def predict_asl_word(sequence):
    """
    Predict the American Sign Language (ASL) word from an image stream.
    """
    predicted_word = None
    predicted_accuracy = 0.0

    # Predict the word
    prediction = model.predict(sequence)
    #print(prediction)
    predicted_label = np.argmax(prediction, axis=-1)
    #print(predicted_label)

    # Get the predicted word
    predicted_word = word_index.get(predicted_label[0], "Unknown")

    # Calculate accuracy of the predicted word
    predicted_accuracy = round(prediction[0][predicted_label[0]] * 100, 2)

    return predicted_word, predicted_accuracy


if __name__ == '__main__':
    video_path = "../raw_data/gabriel_2.MOV"
    cap = cv2.VideoCapture(video_path)
    #print(cap)

    # Initialize MediaPipe modules for hand detection
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils

    # Parameters
    max_sequence_length = 100  # Set a maximum sequence length
    feature_size = 126  # Assuming 21 hand landmarks * 3 = 63
    rolling_window_size = 100  # Set the size for the rolling window

    hand_landmarks_deque = deque(maxlen=rolling_window_size)
    frames = []
    prediction_made = False
    predicted_word = ""
    predicted_accuracy = 0.0
    label = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        hand_results = hands.process(frame_rgb)
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                # Extract x, y, z coordinates for each landmark
                hand_landmarks.extend([lm.x for lm in hand_landmarks_obj.landmark])
                hand_landmarks.extend([lm.y for lm in hand_landmarks_obj.landmark])
                hand_landmarks.extend([lm.z for lm in hand_landmarks_obj.landmark])
            # If only one hand is detected, fill the rest with 63 zeros
            if len(hand_results.multi_hand_landmarks) == 1:
                hand_landmarks.extend([0] * (21 * 3))

            # Draw hand landmarks on the frame
            for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks_obj, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            # Fill with 126 zeros if no hands are detected
            hand_landmarks = [0] * (21 * 3 * 2)

        if len(hand_landmarks) == feature_size:
            hand_landmarks_deque.append(hand_landmarks)

    # If the video is shorter than rolling_window_size, fill the deque with zeros until it reaches the required length
    while len(hand_landmarks_deque) < rolling_window_size:
        hand_landmarks_deque.append([0] * feature_size)

    # If the rolling window is full, make a prediction
    if len(hand_landmarks_deque) == rolling_window_size and not prediction_made:
        sequence = np.array(hand_landmarks_deque, dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        #print(sequence[0][37])
        print('----------------------------')

        # Save hand_landmarks_deque to a JSON file
        json_output_path = os.path.join(ROOT_PATH, 'hand_landmarks.json')
        with open(json_output_path, 'w') as json_file:
            json.dump(list(hand_landmarks_deque), json_file)
        print(f"Hand landmarks saved to {json_output_path}")

        label, confidence = predict_asl_word(sequence)
        if label:
            print(f"Predicted ASL Word: {label} with {confidence:.2f}% confidence")
        else:
            print("No word detected in the video.")

    # Release resources
    cap.release()
    #cv2.destroyAllWindows()
