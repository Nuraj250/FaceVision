import os
import cv2
from datetime import datetime, timedelta
import csv

# Cache to prevent repeated logs
last_logged = {}

def draw_boxes(image, boxes, names):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 10 if top - 10 > 10 else top + 30
        cv2.putText(image, name, (left + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def draw_datetime(image):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, now, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image

def log_recognition(name, confidence, frame=None):
    global last_logged
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/recognitions.csv"

    # Avoid logging duplicates within 5 minutes
    now = datetime.now()
    if name in last_logged:
        if now - last_logged[name] < timedelta(minutes=5):
            return
    last_logged[name] = now

    # Save thumbnail
    if frame is not None:
        thumb_path = f"logs/{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        thumbnail = cv2.resize(frame, (200, 200))
        cv2.imwrite(thumb_path, thumbnail)

    # Log to CSV
    exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Timestamp", "Name", "Confidence (%)", "Thumbnail"])
        writer.writerow([
            now.strftime("%Y-%m-%d %H:%M:%S"),
            name,
            f"{confidence:.2f}",
            thumb_path if frame is not None else ""
        ])

import yagmail

# Only if you want email alerts
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver_email@gmail.com"

yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)

def send_email_alert(name, confidence):
    subject = f"FaceVision Alert: {name} recognized"
    body = f"{name} was recognized with {confidence:.2f}% confidence."
    yag.send(to=EMAIL_RECEIVER, subject=subject, contents=body)
