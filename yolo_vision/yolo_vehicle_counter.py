from ultralytics import YOLO
import cv2
from collections import defaultdict
import requests
import json

model = YOLO("yolov8n.pt")
video_files = ["eg1.mp4", "eg2.mp4", "eg3.mp4", "eg4.mp4"]
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

lane_vehicle_counts = {}

DIST_THRESHOLD = 50

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

for idx in range(len(video_files)):
    video_path = video_files[idx]
    lane_number = idx + 1
    cap = cv2.VideoCapture(video_path)

    tracked_centroids = defaultdict(list)
    class_counts = defaultdict(int)
    window_name = f"Lane {lane_number}"

    print(f"\nProcessing {window_name} - {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    centroid = (cx, cy)

                    is_new = True
                    for old_centroid in tracked_centroids[cls_name]:
                        if euclidean_distance(centroid, old_centroid) < DIST_THRESHOLD:
                            is_new = False
                            break

                    if is_new:
                        tracked_centroids[cls_name].append(centroid)
                        class_counts[cls_name] += 1

                    # Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        total_count = sum(class_counts.values())
        cv2.putText(frame, f"Total: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        y_offset = 80
        for cls in vehicle_classes:
            count = class_counts[cls]
            cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 30

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    lane_vehicle_counts[f"Lane {lane_number}"] = dict(class_counts)

print("\n--- Final Vehicle Count (Per Class) ---")
for lane, class_count in lane_vehicle_counts.items():
    print(f"{lane}: {class_count}")

print("\n--- Sending to Backend ---")
url = "http://localhost:8080/api/traffic/update-counts"

try:
    response = requests.post(url, json=lane_vehicle_counts)
    if response.status_code == 200:
        signal_timings = response.json()
        print("\n--- Signal Timings from Backend ---")
        for lane, time in signal_timings.items():
            print(f"{lane}: {time} seconds green light")
    else:
        print(f"Failed to get signal timings. Status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error connecting to backend: {e}")