import cv2
from deepface import DeepFace
from utils import draw_boxes
import os

DATASET_DIR = "dataset"

def recognize_camera():
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.find(img_path=frame, db_path=DATASET_DIR, enforce_detection=False)
            if result[0].shape[0] > 0:
                identity = os.path.basename(os.path.dirname(result[0]['identity'][0]))
                frame = draw_boxes(frame, [[0, frame.shape[1], frame.shape[0], 0]], [identity])
            else:
                frame = draw_boxes(frame, [[0, frame.shape[1], frame.shape[0], 0]], ["Unknown"])
        except Exception as e:
            print(f"[ERROR] {e}")

        cv2.imshow("FaceVision (DeepFace) - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_camera()
