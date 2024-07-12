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
    cv2.namedWindow("Inference",cv2.WINDOW_NORMAL)
    cv2.imshow("Inference", annotated_image)

    key = cv2.waitKey(0)  # Wait for a key press to close the image window

    if key == ord('q'):
        cv2.destroyAllWindows()
# Close the display window
cv2.destroyAllWindows()
