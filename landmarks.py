import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from tqdm import tqdm
from augmentation import normalize_frame, augment_frame

video_folder = "data/ASL_Citizen/ASL_Citizen/videos"

output_folder = "data/output"
os.makedirs(output_folder, exist_ok=True)
output_npy = "data/output/npy"
os.makedirs(output_npy, exist_ok=True)
summary_folder = os.path.join(output_folder, "summary")
os.makedirs(summary_folder, exist_ok=True)
output_train_file = os.path.join(output_npy, "train_landmarks.npy")
processed_gloss_folder = os.path.join(output_folder, "processed_glosses")
os.makedirs(processed_gloss_folder, exist_ok=True)

train_df = pd.read_csv("data/ASL_Citizen/ASL_Citizen/splits/train.csv")
train_gloss_counts = pd.read_csv(os.path.join(summary_folder, "train_gloss_summary.csv"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_and_save_images(video_path, gloss, sample_index):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    gloss_folder = os.path.join(processed_gloss_folder, gloss)
    os.makedirs(gloss_folder, exist_ok=True)

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = normalize_frame(frame)
        frame = augment_frame(frame, flip=True, rotate=True, brightness=True)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            resized_frame = cv2.resize(frame, (256, 256))
            image_path = os.path.join(gloss_folder, f"{sample_index}_{frame_index}.jpg")
            cv2.imwrite(image_path, resized_frame)
            frame_index += 1

    cap.release()
    return np.array(landmarks_list)

def process_and_save_landmarks(dataframe, output_file, gloss_counts, max_glosses=None):
    all_landmarks = []

    gloss_groups = dataframe.groupby("Gloss")
    if max_glosses is not None:
        gloss_groups = list(gloss_groups)
        np.random.shuffle(gloss_groups)
        gloss_groups = gloss_groups[:max_glosses]

    gloss_iterator = tqdm(gloss_groups, desc="OVERALL PROCESS")

    for gloss, group in gloss_iterator:
        gloss_landmarks = []

        video_count = gloss_counts[gloss_counts["Gloss"] == gloss]["Video Count"].values[0]
        replication_factor = max(1, (20 + video_count - 1) // video_count)

        video_iterator = tqdm(group.iterrows(), total=len(group), desc=f"Processing videos for {gloss}", leave=False)

        for _, row in video_iterator:
            video_file = os.path.join(video_folder, row["Video file"])
            asl_lex_code = row["ASL-LEX Code"]

            for sample_index in range(replication_factor):
                landmarks = extract_landmarks_and_save_images(video_file, gloss, sample_index)
                if landmarks.size > 0:
                    gloss_landmarks.append({
                        "asl_lex_code": asl_lex_code,
                        "landmarks": landmarks
                    })

        if gloss_landmarks:
            all_landmarks.append({
                "gloss": gloss,
                "videos": gloss_landmarks
            })

    np.save(output_file, all_landmarks)
    print(f"Upsampled video counts saved to {output_file}")

time.sleep(2)
user_input = input("\n\n\n\nEnter the number of glosses to process (or 'all' to process all): ")

try:
    max_glosses = int(user_input)
except ValueError:
    max_glosses = None if user_input.lower() == 'all' else 0

print("\n\nProcessing training data:\n")
process_and_save_landmarks(train_df, output_train_file, train_gloss_counts, max_glosses)
