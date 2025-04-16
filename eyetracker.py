import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialisation Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Paramètres
CALIBRATION_X = 1.5
CALIBRATION_Y = 1.2
SMOOTHING_FACTOR = 0.4
BLINK_THRESHOLD = 0.25            # ← sensibilité (plus haut = moins sensible)
BLINK_FRAMES_THRESHOLD = 6       # ← plus haut = moins sensible

# Variables de suivi
prev_x, prev_y = 0, 0
tracking_active = True
blink_counter_left = blink_counter_right = 0
left_eye_closed_frames = 0
right_eye_closed_frames = 0

# Points des yeux
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_EAR(eye):
    A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
    B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
    C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
    return (A + B) / (2.0 * C)

def get_eye_center(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    return np.mean(pts, axis=0)

def map_to_screen(x, y, w, h):
    screen_w, screen_h = pyautogui.size()
    sx = int((x / w) * screen_w * CALIBRATION_X)
    sy = int((y / h) * screen_h * CALIBRATION_Y)
    return max(0, min(sx, screen_w - 1)), max(0, min(sy, screen_h - 1))

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
window_name = "Eye Tracker"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        # Clignement œil gauche
        left_EAR = compute_EAR([face[i] for i in LEFT_EYE])
        if left_EAR < BLINK_THRESHOLD:
            left_eye_closed_frames += 1
        else:
            if left_eye_closed_frames >= BLINK_FRAMES_THRESHOLD:
                pyautogui.click()
                blink_counter_left += 1
            left_eye_closed_frames = 0

        # Clignement œil droit
        right_EAR = compute_EAR([face[i] for i in RIGHT_EYE])
        if right_EAR < BLINK_THRESHOLD:
            right_eye_closed_frames += 1
        else:
            if right_eye_closed_frames >= BLINK_FRAMES_THRESHOLD:
                pyautogui.click(button='right')
                blink_counter_right += 1
            right_eye_closed_frames = 0

        # Déplacement du curseur
        if tracking_active:
            left_center = get_eye_center(face, LEFT_EYE)
            right_center = get_eye_center(face, RIGHT_EYE)
            eye_x = (left_center[0] + right_center[0]) / 2 * frame.shape[1]
            eye_y = (left_center[1] + right_center[1]) / 2 * frame.shape[0]

            screen_x, screen_y = map_to_screen(eye_x, eye_y, frame.shape[1], frame.shape[0])
            screen_x = int(SMOOTHING_FACTOR * prev_x + (1 - SMOOTHING_FACTOR) * screen_x)
            screen_y = int(SMOOTHING_FACTOR * prev_y + (1 - SMOOTHING_FACTOR) * screen_y)
            prev_x, prev_y = screen_x, screen_y
            pyautogui.moveTo(screen_x, screen_y)

        # Affichage
        cv2.putText(frame, f"CLICS G: {blink_counter_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CLICS D: {blink_counter_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
        cv2.putText(frame, f"TRACKING: {'ON' if tracking_active else 'OFF'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord("p"):
        tracking_active = not tracking_active

cap.release()
cv2.destroyAllWindows()
