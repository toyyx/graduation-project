import sys
import cv2
import numpy as np
sys.path.append("..")
from segment_anything import sam_model_registry_new, SamPredictor_new

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry_new[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor_new(sam)

image = cv2.imread('images/truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)