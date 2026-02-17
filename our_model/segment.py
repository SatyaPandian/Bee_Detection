from ultralytics import SAM
import supervision as sv
import cv2
model = SAM('sam_b.pt')
image = cv2.imread("/Users/satyapandian/Documents/YoloTest/Bee_Detection/our_model/honey_bee.jpeg")
results = model("/Users/satyapandian/Documents/YoloTest/Bee_Detection/our_model/honey_bee.jpeg")
print(results.boxes)

