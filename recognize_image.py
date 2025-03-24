import os
import cv2
from deepface import DeepFace
from utils import draw_boxes

TEST_IMAGES_DIR = "test_images"
DATASET_DIR = "dataset"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def recognize_images():
    for image_name in os.listdir(TEST_IMAGES_DIR):
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        print(f"[INFO] Processing {image_path}...")

        result = DeepFace.find(img_path=image_path, db_path=DATASET_DIR, enforce_detection=False)
        image = cv2.imread(image_path)

        if result[0].shape[0] > 0:
            # Best match
            identity = os.path.basename(os.path.dirname(result[0]['identity'][0]))
            image = draw_boxes(image, [[0, image.shape[1], image.shape[0], 0]], [identity])
            print(f"[MATCH] Found: {identity}")
        else:
            image = draw_boxes(image, [[0, image.shape[1], image.shape[0], 0]], ["Unknown"])
            print("[NO MATCH] No known face found.")

        output_path = os.path.join(OUTPUT_DIR, f"recognized_{image_name}")
        cv2.imwrite(output_path, image)
        print(f"[INFO] Saved result to {output_path}")

if __name__ == "__main__":
    recognize_images()
