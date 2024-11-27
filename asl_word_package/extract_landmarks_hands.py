import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from multiprocessing import Pool
import warnings
import logging
import absl.logging
import sys

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*inference_feedback_manager.*")

# Suppress specific log messages
logging.getLogger('mediapipe').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.get_absl_handler().use_absl_log_file('absl_logging', '/tmp')  # Redirect absl logs to file to suppress CLI warnings

# Directory containing videos
VIDEOS_DIR = "data/videos"

# Parameters
max_sequence_length = 100  # Set a maximum sequence length
feature_size = 126  # Assuming 21 hand landmarks_data * 3 = 63

# Function to collect hand landmarks from video
def collect_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    hand_landmarks_list = []

    hands = mp.solutions.hands.Hands()  # Initialize MediaPipe Hands only once for efficiency

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
        else:
            # Fill with 126 zeros if no hands are detected (assuming 21 landmarks per hand with x, y, z for each)
            hand_landmarks = [0] * (21 * 3 * 2)
        
        hand_landmarks_list.append(hand_landmarks)

    cap.release()
    hands.close()  # Close MediaPipe Hands to release resources
    return hand_landmarks_list

# Function to process each video file
def process_video(video_file):
    try:
        if video_file.endswith(".mp4"):
            # Extract the word from the video file name
            video_path = os.path.join(VIDEOS_DIR, video_file)
            word = video_file.split('-')[1].split('.mp4')[0].replace('seed', '').strip()
            #word = ''.join(filter(str.isalpha, video_file.split('-')[1].split('.mp4')[0])).replace('seed', '').strip()
            print(video_path)

            # Collect hand landmarks for the entire video
            hand_landmarks = [landmark for landmark in collect_landmarks(video_path) if len(landmark) > 0]

            # Return the landmarks and the associated word
            return {
                'word': word,
                'hand_landmarks': hand_landmarks
            }

    except Exception as e:
        print(f"Error processing video {video_file}: {e}")
        return None

# Main function to process videos and save results
def main():
    counter = 0
    landmarks_dataset = []

    # Use multiprocessing Pool to process videos
    with Pool() as pool:
        all_videos = [video_file for video_file in os.listdir(VIDEOS_DIR) if video_file.endswith(".mp4")]
        for processed_data in pool.imap_unordered(process_video, all_videos):
            if processed_data:
                counter += 1
                landmarks_dataset.append(processed_data)
                print(f"Processed: {counter}")

    # Save all landmarks to a pickle file at the end
    pkl_filename = "word_landmarks_hand_large_multiple.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(landmarks_dataset, f)
    print(f"Saved all landmarks to {pkl_filename}")

    print("Landmarks extraction complete.")

# Run the main function
if __name__ == "__main__":
    print('Starting')
    main()

