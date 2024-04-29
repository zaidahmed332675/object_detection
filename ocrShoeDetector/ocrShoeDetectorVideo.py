from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Initialize the OCR reader
reader = Reader(['en'], gpu=False)

# Load a pretrained YOLOv8n model
model = YOLO('/content/detector/8.pt')  # Specify the device to 'cpu' if no GPU is available

# Define the video capture device (e.g. a camera or video file)
cap = cv2.VideoCapture('/content/detector/shoeVideo.mp4')  # Replace with your video file or camera index
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform object detection
    results = model(rgb_frame)
    result = results[0]

    # print(results[0].boxes, 'all results')
    # break

    # Process results list
    for box in results[0].boxes.xyxy:
        # print(result, 'single result')
        # break

        # Accessing bounding boxes in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract the region of interest based on bounding box coordinates
        roi = result.orig_img[y1:y2, x1:x2]

        # Perform OCR on the extracted region
        ocr_result = reader.readtext(roi)
        box_text = ""

        # Optionally draw OCR results on the original image for visualization
        for ocrResult in ocr_result:
            bbox, text, score = ocrResult
            if score > 0.3:  # Confidence threshold
                box_text += text + ", "
        
        textInlinePadding = 5
        fontScaling = 0.6
        thinkness = 2

        text_width, _ = cv2.getTextSize(box_text.rstrip(", "), cv2.FONT_HERSHEY_SIMPLEX, fontScaling, thinkness)[0]
        text_width = (textInlinePadding * 2) + text_width + x1
        text_width = min(text_width, result.orig_img.shape[1] - 1)  # Avoid going out of image width

        cv2.rectangle(frame, (x1, y1 - 35), (text_width, y1), (200, 50, 100), -1)
        cv2.putText(frame, box_text.rstrip(", "), (x1 + textInlinePadding, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScaling, (255, 255, 255), thinkness)


        print('box loop')
        # break

    # cv2_imshow(frame)
    out.write(frame)
    
    print('frame-',cap.get(cv2.CAP_PROP_POS_FRAMES))
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= 40:
        print('frames reached limit')
        break
    print('while loop')
        # break
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()