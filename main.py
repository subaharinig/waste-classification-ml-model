import cv2
import threading
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

# Load classifier (must be trained binary Recyclable/Non-Recyclable)
model = YOLO("yolov8n-cls.pt")


# Mediapipe hand detector setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global flags
processing = False
result_text = ""
frame_to_display = None

def classify_hand_object(cropped_img):
    global result_text, processing, frame_to_display
    # Save temporary cropped image
    cv2.imwrite("temp_crop.jpg", cropped_img)
    # Run classification
    results = model("temp_crop.jpg", verbose=False)
    pred = results[0].probs.top1  # class index
    class_name = "Recyclable" if pred == 0 else "Non-Recyclable"
    result_text = class_name
    frame_to_display = cropped_img.copy()
    processing = False


def main():
    global processing, frame_to_display, result_text
    cap = cv2.VideoCapture(0)

    while True:
        if not processing:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get bounding box from hand landmarks
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min = int(min(x_coords) * w) - 20
                    y_min = int(min(y_coords) * h) - 20
                    x_max = int(max(x_coords) * w) + 20
                    y_max = int(max(y_coords) * h) + 20

                    # Clamp values to frame size
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)

                    cropped = frame[y_min:y_max, x_min:x_max]

                    # Pause and start classification thread
                    processing = True
                    threading.Thread(target=classify_hand_object, args=(cropped,)).start()

                    # Draw box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    break

            # Show live feed
            cv2.imshow("Live Feed", frame)

        else:
            # Show classification result
            if frame_to_display is not None:
                display_img = frame_to_display.copy()
                cv2.putText(display_img, f'Class: {result_text}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result_text == "Recyclable" else (0, 0, 255), 2)
                cv2.imshow("Classification Result", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
