import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load YOLOv8 model
# model = YOLO('yolov8n6.pt')
model = YOLO('yolov8n.pt')

# Define camera calibration parameters
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
dist_coeffs = np.array([0, 0, 0, 0, 0])

# Define speed calculation parameters
fps = 30  # frames per second
pixel_density = 3.2  # pixels per meter

# Define the video capture device (e.g. a camera or video file)
cap = cv2.VideoCapture('/content/drive/MyDrive/yolo_ultra/TrafficCamera.mp4')  # Replace with your video file or camera index

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fov = 60

# Define the virtual line (x1, y1, x2, y2)
virtual_line = [(0, frame_height // 2), (frame_width, frame_height // 2)] # Adjust these coordinates to your needs

# Set the frame position to 10 seconds (300 frames at 30 fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Define Kalman Filter parameters (adjust based on your needs)
# kalman = cv2.KalmanFilter(4, 2, 0)
# kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                  #  [0, 1, 0, 0]], dtype=np.float32)
# kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                #  [0, 1, 0, 0],
                                #  [0, 0, 1, 0],
                                #  [0, 0, 0, 1]], dtype=np.float32) * 0.01

# Initialize Kalman Filter state vector
# state = np.zeros((4, 1), dtype=np.float32)

def track_car(frame, bounding_box):
  """Tracks a car using Kalman Filter based on the provided bounding box.

  Args:
      frame: The current frame from the video.
      bounding_box: A tuple containing (x1, y1, x2, y2) coordinates of the bounding box.

  Returns:
      A tuple containing the updated bounding box and the Kalman Filter state vector.
  """
  global kalman, state

  # Predict the state
  prediction = kalman.predict()

  # Extract center of the bounding box
  center_x = int((bounding_box[0] + bounding_box[2]) / 2)
  center_y = int((bounding_box[1] + bounding_box[3]) / 2)

  # Update the Kalman Filter with the new center position
  measurement = np.array([center_x, center_y], dtype=np.float32).reshape(2, 1)
  kalman.correct(measurement)

  # Extract predicted state for drawing the bounding box
  predicted_x = int(prediction[0])
  predicted_y = int(prediction[1])

  # Update the bounding box based on Kalman Filter prediction
  updated_bounding_box = (predicted_x - 25, predicted_y - 25, predicted_x + 25, predicted_y + 25)

  # Draw the predicted bounding box (optional)
  cv2.rectangle(frame, updated_bounding_box, (0, 255, 0), 2)

  return updated_bounding_box, state

prev_x = 0  # Initialize with any value (will be overwritten in the first iteration)
prev_y = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)

    # print(results,' whole')
    # break

    # Loop through detected objects (cars)
    for data in results[0].boxes.data:
        x, y, x2, y2, confidence, class_id = data

        print(x, y, x2, y2, confidence, class_id, virtual_line, 'confidense and classId')

        # Filter out non-car detections
        if confidence > 0.5 and class_id == 2:
            # Check if the car touches the virtual line
            # if (y2) > virtual_line[0][1] and (y2) < virtual_line[1][1] and (x2) > virtual_line[0][0] and (x2) < virtual_line[1][0]:
            if (y2 - y) > (x2 - x):
                # updated_bounding_box, state = track_car(frame, bounding_box=(x, y, x2, y2))

                # Define parameters (might require adjustments)
                # dt = 1 / fps  # Time difference between frames (assuming constant FPS)
                # pixel_to_meter = 1 / pixel_density  # Conversion from pixels to meters

                # Calculate speed based on Kalman Filter prediction
                # predicted_x = state[0][0]  # Assuming state[0] contains predicted x-position
                # predicted_y = state[1][0]  # Assuming state[1] contains predicted y-position

                # Assuming constant velocity model (adjust based on your Kalman Filter setup)
                # velocity_x = (predicted_x - prev_x) / dt * pixel_to_meter  # Velocity in meters/second (x-direction)
                # velocity_y = (predicted_y - prev_y) / dt * pixel_to_meter  # Velocity in meters/second (y-direction)

                # prev_x = predicted_x
                # prev_y = predicted_y

                # Calculate speed magnitude (assuming horizontal movement is dominant)
                # speed = np.sqrt(velocity_x**2 + velocity_y**2)  # Speed in meters/second
                # speed_kmph = speed * 3.6  # Convert to kilometers per hour

                # cv2.rectangle(frame, updated_bounding_box, (0, 255, 0), 2)  # Draw green rectangle around tracked car
                # cv2.putText(frame, f"{speed:.2f} km/h", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Calculate the speed
                # print('executing bro', (y2) > virtual_line[0][1] and (y2) < virtual_line[1][1] and (x2) > virtual_line[0][0] and (x2) < virtual_line[1][0])
                distance = (x2 - x) * pixel_density / (2 * np.tan(camera_fov * np.pi / 180))
                speed_kmph = (distance / fps) * 3.6  # Convert to km/h


                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)


                # Draw corner points with x and y values
                corner_points = [(int(x), int(y)), (int(x2), int(y)), (int(x2), int(y2)), (int(x), int(y2))]

                for i, point in enumerate(corner_points):
                    print(point,'cheking point')
                    cv2.circle(frame, point, 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{speed_kmph:.2f} km/h", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.putText(frame, f'x{i+1}: {point[0]} y{i+1}: {point[1]}', (point[0] + 10, point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw the bounding box and speed on the frame
                # cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.putText(frame, f"{speed_kmph:.2f} km/h", (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw the virtual line
    # cv2.line(frame, virtual_line[0], virtual_line[1], (0, 255, 0), 2)

    # Display the output and save it to the video file
    cv2_imshow(frame)
    out.write(frame)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= 300:
        break
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()