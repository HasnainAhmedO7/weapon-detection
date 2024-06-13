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
    count+=1
    if count % 2!=0:
        continue
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,conf=0.15,iou=0.5)

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