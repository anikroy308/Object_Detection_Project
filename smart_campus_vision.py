"""
Smart Campus Vision
-------------------
A computer vision-based campus monitoring system using Python,
OpenCV, and YOLO.

Features:
- Real-time object detection
- Object tracking
- Entry and exit monitoring
- Occupancy estimation
- Movement analysis
- Approximate posture estimation

Author: Anik Roy
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIGURATION =================
CAMERA_INDEX = 1
MODEL_NAME = "yolov8n.pt"

BLUR_THRESHOLD = 100
LIGHT_THRESHOLD = 100
MOTION_THRESHOLD = 800

LINE_POSITION_RATIO = 0.50
CROSSING_COOLDOWN = 15

# Load YOLO model
model = YOLO(MODEL_NAME)

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try CAMERA_INDEX = 0.")

# Background motion detector
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

previous_positions = {}
last_crossing_frame = {}
unique_entry_ids = set()

occupancy = 0
frame_number = 0

try:
    while True:

        # 1. Capture frame
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Frame capture failed. Stopping.")
            break

        frame_number += 1
        height, width = frame.shape[:2]
        line_y = int(height * LINE_POSITION_RATIO)

        # 2. Motion detection
        mask = bg_subtractor.apply(frame)

        _, mask = cv2.threshold(
            mask, 200, 255, cv2.THRESH_BINARY
        )

        motion_pixels = cv2.countNonZero(mask)
        motion_detected = motion_pixels > MOTION_THRESHOLD

        # 3. Blur detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()

        frame_clear = blur_score >= BLUR_THRESHOLD

        # 4. Approximate lighting detection
        brightness = float(np.mean(gray))
        lights_on = brightness >= LIGHT_THRESHOLD

        # 5. Person detection and tracking
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )

        people = 0
        seated_estimate = 0
        standing_moving_estimate = 0

        boxes = results[0].boxes if results else None

        if boxes is not None and len(boxes) > 0:

            coords = boxes.xyxy.cpu().numpy()

            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(coords)

            for box, track_id in zip(coords, ids):

                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                people += 1
                moving = False
                previous = None

                # 6. Tracking and line crossing
                if track_id is not None:

                    previous = previous_positions.get(track_id)

                    if previous is not None:

                        px, py = previous

                        distance = np.hypot(cx - px, cy - py)
                        moving = distance > 12

                        cooldown_ok = (
                            frame_number -
                            last_crossing_frame.get(track_id, -999)
                            > CROSSING_COOLDOWN
                        )

                        if cooldown_ok:

                            # Top to bottom = entry
                            if py < line_y <= cy:
                                occupancy += 1
                                unique_entry_ids.add(track_id)
                                last_crossing_frame[track_id] = frame_number

                            # Bottom to top = exit
                            elif py >= line_y > cy:
                                occupancy = max(0, occupancy - 1)
                                last_crossing_frame[track_id] = frame_number

                    # Update history every frame
                    previous_positions[track_id] = (cx, cy)

                # 7. Approximate posture estimation
                box_height = max(1, y2 - y1)
                box_width = max(1, x2 - x1)

                aspect_ratio = box_height / box_width

                if not moving and aspect_ratio < 1.25:
                    seated_estimate += 1
                    posture = "Possible Seated"
                else:
                    standing_moving_estimate += 1
                    posture = "Standing/Moving"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"ID:{track_id} {posture}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # 8. Dashboard status
        occupancy = max(0, occupancy)

        room_status = "Occupied" if people > 0 else "Empty"
        motion_status = "Motion Detected" if motion_detected else "No Motion"
        quality_status = "Clear" if frame_clear else "Blurry"
        light_status = "Lights ON" if lights_on else "Lights OFF"

        empty_room_lights_alert = (people == 0 and lights_on)

        # Draw crossing line
        cv2.line(
            frame,
            (0, line_y),
            (width, line_y),
            (255, 0, 0),
            2
        )

        dashboard = [
            f"Present Now: {people}",
            f"Estimated Occupancy: {occupancy}",
            f"Unique Entry IDs: {len(unique_entry_ids)}",
            f"Possible Seated: {seated_estimate}",
            f"Standing/Moving: {standing_moving_estimate}",
            f"Motion: {motion_status}",
            f"Room: {room_status}",
            f"Frame: {quality_status} ({blur_score:.1f})",
            f"Lighting: {light_status} ({brightness:.1f})"
        ]

        if empty_room_lights_alert:
            dashboard.append("ALERT: Lights ON, room appears empty")

        # Display dashboard
        for i, text in enumerate(dashboard):

            cv2.putText(
                frame,
                text,
                (15, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        # Display video and motion mask
        cv2.imshow("iCloudEMS Smart Campus Vision", frame)
        cv2.imshow("Motion Mask", mask)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Program closed.")
