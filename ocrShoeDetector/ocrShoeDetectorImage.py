from ultralytics import YOLO
from easyocr import Reader
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Initialize the OCR reader
reader = Reader(['en'], gpu=False)

# Load a pretrained YOLOv8n model
model = YOLO('/content/detector/8.pt')  # Specify the device to 'cpu' if no GPU is available

# Run batched inference on a list of images
results = model(['/content/detector/shoeImg.jpeg'])  # return a list of Results objects

# Process results list
for result in results:
    # Accessing bounding boxes in [x1, y1, x2, y2] format
    boxes = result.boxes.xyxy  # This will give you the bounding boxes in the correct format

    # Process each detected box
    for box in boxes:
        # print(box, 'box is here')
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract the region of interest based on bounding box coordinates
        roi = result.orig_img[y1:y2, x1:x2]
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Binarize
        
        # cv2_imshow(roi)

        # Perform OCR on the extracted region
        ocr_result = reader.readtext(roi)
        # print(ocr_result, 'reayOcr result')

        # testing rectangle genertion with custom coords
        # bboxx = [[11, 9], [59, 9], [59, 29], [11, 29]]
        # cv2.rectangle(result.orig_img, tuple(bboxx[0]), tuple(bboxx[2]), (255, 0, 0), 2)
        # break

        box_text = ""
        # Optionally draw OCR results on the original image for visualization
        for ocrResult in ocr_result:
            bbox, text, score = ocrResult
            
            # here is the issue in bbox coords
            print(bbox, text, score)
            # break

            if score > 0.3:  # Confidence threshold
                box_text += text + ", "
                # top_left = (int(box[0][0]), int(box[0][1]))
                # bottom_right = (int(box[2][0]), int(box[2][1]))
                # top_left = (int(bbox[0][0]), int(bbox[0][1]))
                # bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
                # cv2.rectangle(result.orig_img, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)
                # cv2.putText(result.orig_img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # print(text, 'text just before print')
                # cv2.putText(result.orig_img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        textInlinePadding = 5
        fontScaling = 0.6
        thinkness = 2

        text_width, _ = cv2.getTextSize(box_text.rstrip(", "), cv2.FONT_HERSHEY_SIMPLEX, fontScaling, thinkness)[0]
        text_width = (textInlinePadding * 2) + text_width + x1
        text_width = min(text_width, result.orig_img.shape[1] - 1)  # Avoid going out of image width

        cv2.rectangle(result.orig_img, (x1, y1 - 35), (text_width, y1), (200, 50, 100), -1)
        cv2.putText(result.orig_img, box_text.rstrip(", "), (x1 + textInlinePadding, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScaling, (255, 255, 255), thinkness)
        # break

    # Display processed image to screen and save to disk
    result.show()  # display to screen
    cv2.imwrite('result.jpg', cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR))  # Save the image with OCR results