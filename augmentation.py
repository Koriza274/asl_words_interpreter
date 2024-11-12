import cv2
import numpy as np

def normalize_frame(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)

    frame = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)

    normalized_frame = frame / 255.0
    return (normalized_frame * 255).astype(np.uint8)

def augment_frame(frame, flip=True, rotate=True, brightness=True):
    if flip:
        frame = cv2.flip(frame, 1)

    if rotate:
        angle = np.random.uniform(-15, 15)
        h, w = frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        frame = cv2.warpAffine(frame, rotation_matrix, (w, h))

    if brightness:
        value = np.random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        frame = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

    return frame
