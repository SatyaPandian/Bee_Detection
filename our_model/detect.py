from inference import get_model
import supervision as sv
import cv2

image1 = "bee_vs_wasp.jpeg"
image2 = "honey_bee.jpeg"
image3 = "bees.jpeg"

for frame, reg in cv2.VideoCapture(0):
    

image = cv2.imread(image3)
model = get_model(model_id="honey-bee-detection-model-zgjnb/2", api_key="O5Kea7msXygAFNjhnOhq")
results = model.infer(image)[0]
detections = sv.Detections.from_inference(results)
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)
sv.plot_image(annotated_image)