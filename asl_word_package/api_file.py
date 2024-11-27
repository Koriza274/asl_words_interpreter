from fastapi import FastAPI, Request, HTTPException
import numpy as np
import cv2
import base64
import mediapipe as mp
from collections import deque
from asl_word_package.word_interpreter import predict_asl_word
from PIL import Image
import io
import imageio

# FastAPI instance
app = FastAPI()

# Initialize MediaPipe for hand detection
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Parameters
feature_size = 126
rolling_window_size = 100

@app.get("/")
def root():
    return {'greeting': "ready for ASL word prediction!"}

@app.post("/predict_word")
async def predict_word(request: Request):
    try:
        data = await request.json()
        frames_data = data.get("frames")

        if not frames_data:
            raise HTTPException(status_code=400, detail="No frames provided in the request.")

        hand_landmarks_deque = deque(maxlen=rolling_window_size)
        frames_deque = deque(maxlen=rolling_window_size)

        # Process each frame
        for frame_data in frames_data:
            # Decode the base64 encoded frame back to an image
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process frame using MediaPipe to extract landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            hand_landmarks = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                    hand_landmarks.extend([lm.x for lm in hand_landmarks_obj.landmark])
                    hand_landmarks.extend([lm.y for lm in hand_landmarks_obj.landmark])
                    hand_landmarks.extend([lm.z for lm in hand_landmarks_obj.landmark])
                    #drawing_utils.draw_landmarks(frame, hand_landmarks_obj, mp.solutions.hands.HAND_CONNECTIONS)
                    # Draw landmarks on the frame with customized thickness and radius
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks_obj,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),  # Customize color, thickness, and circle radius for landmarks
                        drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)  # Customize color and thickness for connections
                    )
                if len(hand_results.multi_hand_landmarks) == 1:
                    hand_landmarks.extend([0] * (21 * 3))
            else:
                hand_landmarks = [0] * (21 * 3 * 2)

            if len(hand_landmarks) == feature_size:
                hand_landmarks_deque.append(hand_landmarks)
                frames_deque.append(frame)
         # If the video is shorter than rolling_window_size, fill the deque with zeros until it reaches the required length
        while len(hand_landmarks_deque) < rolling_window_size:
            hand_landmarks_deque.append([0] * feature_size)


        # If rolling window is full, make a prediction
        if len(hand_landmarks_deque) == rolling_window_size:
            sequence = np.array(hand_landmarks_deque, dtype=np.float32)
            sequence = np.expand_dims(sequence, axis=0)

            # Call the prediction function from the separate module
            label, confidence = predict_asl_word(sequence)
            hand_landmarks_deque.clear()

            # Create a GIF from the collected frames
            frames = list(frames_deque)
            frames_deque.clear()

            gif_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 240)) for frame in frames]
            gif_buffer = io.BytesIO()
            imageio.mimsave(gif_buffer, gif_frames, format='GIF', fps=25, loop=0)
            gif_buffer.seek(0)
            gif_base64 = base64.b64encode(gif_buffer.read()).decode('utf-8')

            if label is None:
                raise HTTPException(status_code=400, detail="No word detected in the video.")

            # Return the prediction result along with the GIF
            return {
                "prediction": str(label),
                "confidence": float(confidence),
                "gif": gif_base64
            }

        return {"prediction": "collecting frames for prediction...", "confidence": 0.0}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
