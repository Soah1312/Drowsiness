import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Drowsiness Detection Constants
EYE_AR_THRESHOLD = 0.25
TILT_THRESHOLD = 15
TIME_THRESHOLD = 3  # Must be drowsy for 3 seconds
BRIGHTNESS_THRESHOLD = 50  # If brightness is below this, apply enhancements
COUNTER_EYES = 0
COUNTER_TILT = 0
start_eye_time = None
start_tilt_time = None


def get_brightness(frame):
    """Calculates the average brightness of the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def apply_low_light_enhancements(frame):
    """Enhances the frame in low-light conditions using CLAHE & Gamma Correction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Gamma correction to enhance dark areas
    gamma = 1.5  # Adjust this value for better brightness
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_gray = cv2.LUT(enhanced_gray, look_up_table)

    return enhanced_gray


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Correct inverted feed

    # Check brightness first
    brightness = get_brightness(frame)
    if brightness < BRIGHTNESS_THRESHOLD:
        frame = cv2.cvtColor(apply_low_light_enhancements(frame), cv2.COLOR_GRAY2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        start_eye_time = None
        start_tilt_time = None
        continue

    for face_landmarks in results.multi_face_landmarks:
        height, width, _ = frame.shape
        landmarks = [(int(l.x * width), int(l.y * height)) for l in face_landmarks.landmark]

        # Eye landmarks
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        HEAD_TILT_POINTS = [10, 152]  # Forehead & Chin


        def calculate_ear(eye_points, landmarks):
            A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
            B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
            C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
            return (A + B) / (2.0 * C)


        left_ear = calculate_ear(LEFT_EYE, landmarks)
        right_ear = calculate_ear(RIGHT_EYE, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        # Calculate head tilt angle
        top_point = landmarks[HEAD_TILT_POINTS[0]]
        bottom_point = landmarks[HEAD_TILT_POINTS[1]]
        angle = np.degrees(np.arctan2(bottom_point[0] - top_point[0], bottom_point[1] - top_point[1]))

        # Eye Drowsiness Detection
        if avg_ear < EYE_AR_THRESHOLD:
            if start_eye_time is None:
                start_eye_time = time.time()
            elif time.time() - start_eye_time >= TIME_THRESHOLD:
                winsound.Beep(1000, 1000)  # Beep if eyes are closed for too long
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            start_eye_time = None

        # Head Tilt Detection
        if abs(angle) > TILT_THRESHOLD:
            if start_tilt_time is None:
                start_tilt_time = time.time()
            elif time.time() - start_tilt_time >= TIME_THRESHOLD:
                winsound.Beep(1000, 1000)  # Beep if head is tilted for too long
                cv2.putText(frame, "HEAD TILT ALERT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            start_tilt_time = None

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
