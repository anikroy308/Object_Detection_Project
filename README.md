# Smart Campus Vision – AI-Based Campus Monitoring System

## Project Overview
Smart Campus Vision is a computer vision-based monitoring system designed to analyze campus activity using camera feeds. It aims to support smart campus environments through real-time object detection, tracking, entry-exit monitoring, and occupancy estimation.

The system processes video streams to detect and track objects/people, analyze movement across designated lines, and estimate occupancy based on detected movement patterns.

## Key Features
- Real-time computer vision-based detection
- Camera feed monitoring using OpenCV
- Object tracking with unique tracking IDs
- Entry and exit detection using virtual crossing lines
- Occupancy estimation
- Movement tracking and cooldown logic to reduce duplicate counting
- Approximate visual/posture analysis based on detected bounding boxes
- Support for multiple camera sources, depending on hardware availability

## Technologies Used
- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- Computer Vision
- Object Tracking

## Project Structure

```text
Smart_Campus_Vision/
│
├── smart_campus_vision.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/anikroy308/Object_Detection_Project.git
cd Object_Detection_Project
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main project file:

```bash
python smart_campus_vision.py
```

Make sure your camera is connected and accessible to the system.

## Working Principle

1. The system captures video frames from the connected camera.
2. Frames are processed using computer vision and the detection model.
3. Detected objects are tracked across frames.
4. Virtual crossing lines are used to identify entry and exit movement.
5. Occupancy is updated based on detected crossing events.
6. Tracking history and cooldown logic help reduce repeated counting.

## Applications
- Campus entry and exit monitoring
- Occupancy estimation in monitored areas
- Smart building monitoring
- Computer vision-based traffic and movement analysis

## Limitations
- Detection performance depends on camera quality, lighting, and visibility.
- Occupancy estimates may not always reflect the actual number of people.
- Camera availability and performance depend on the hardware and configuration.
- Posture estimation is approximate and based on visual detection geometry.

## Author
Anik Roy
