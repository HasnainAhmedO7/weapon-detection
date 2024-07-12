# Real-time Weapon Detection Algorithm

This project demonstrates a real-time weapon detection algorithm using the YOLOv8 model for both video and image inference. The algorithm detects weapons in video frames or static images and visualizes the results.

Installation
Ensure you have the following dependencies installed:

Python 3.6+
OpenCV
ultralytics (for YOLOv8)
You can install the required packages using pip:

bash
Copy code
pip install opencv-python ultralytics
Usage
Video Inference
The following script processes a video file and runs YOLOv8 inference on each frame to detect weapons.

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights/best.pt")
count = 0

# Open the video file
video_path = "assets/videos/gun1.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break
    count += 1
    if count % 2 != 0:
        continue
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.15, iou=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
Image Inference
```

The following script processes an image file and runs YOLOv8 inference to detect weapons.

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights/best.pt")

# Path to the image file
image_path = "assets/images/gun.jpeg"

# Read the image
image = cv2.imread(image_path)

# Check if the image was successfully read
if image is None:
    print(f"Error: Could not read image from {image_path}")
else:
    # Run YOLOv8 inference on the image
    results = model(image, conf=0.15)

    # Visualize the results on the image
    annotated_image = results[0].plot()

    # Display the annotated image
    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    cv2.imshow("Inference", annotated_image)

    key = cv2.waitKey(0)  # Wait for a key press to close the image window
    if key == ord('q'):
        cv2.destroyAllWindows()

# Close the display window
cv2.destroyAllWindows()
```

Model Weights
Make sure to place your YOLOv8 model weights file (best.pt) in the weights directory. Adjust the path in the scripts accordingly if your weights file is located elsewhere.

Notes
Press q to quit the video or image inference display.
Press p to pause the video inference display. Press any key to resume.
