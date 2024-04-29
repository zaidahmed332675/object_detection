import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "/content/yolo_ultra/TrafficCamera.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
max_frames = 300

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        cv2_imshow(cv2.resize(annotated_frame, (480, 480)))

        frame_count += 1

        if frame_count >= max_frames:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()