import cv2
from ultralytics import YOLO
import time
import numpy as np
# Load a model
model = YOLO("bestnamegood.pt")  # load a pretrained model (recommended for training)
# Use the model
#video_path="test1.jpg"
#results = model(video_path,save=True)  # predict on an image

pic= "bus.jpg"
point = (150, 200)  # Example coordinates to check if the coordinate on  a person change this to uv coordinate
results = model(pic)

cap = cv2.VideoCapture(pic)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度

annotated_pic=results[0].plot()

# Get the mask for the person class (class index 24) 
masks = results[0].masks.data
classes = results[0].boxes.cls

# Find masks corresponding to the person class (typically class index 0) 
# This part may vary depending on your model's specifics
person_class_index = 24 #the person class index  
person_masks = [masks[i] for i in range(len(classes)) if classes[i] == person_class_index]

# Create a combined mask for all detected persons
combined_person_mask = np.zeros((height, width), dtype=np.uint8)
for mask in person_masks:
    mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (width, height))
    combined_person_mask = cv2.bitwise_or(combined_person_mask, mask_resized)
# Check if the point is within the person mask
t=0
while(1):
    results = model(pic)
    masks = results[0].masks.data
    classes = results[0].boxes.cls
    annotated_pic=results[0].plot()

    point = (150, 200+t)#moving the point 
    t=t+10
    is_on_person = combined_person_mask[point[1], point[0]] > 0
    print(f"The point {point} is {'on a person' if is_on_person else 'not on a person'} in the image.") 
    cv2.circle(annotated_pic, point, radius=5, color=(255, 255, 255), thickness=-1)
    cv2.line(annotated_pic, [150,400], [150,800], color=(255, 255, 0), thickness=2)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

    cv2.resizeWindow("output", width, height)
    cv2.imshow('output', annotated_pic)
    
    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
    
    if key == ord('q'):#press c to quit
        break
    elif key == ord('c'):#press c to move the dot
        continue
   

# Release everything when done

cv2.destroyAllWindows()
