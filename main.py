import cv2
import numpy as np

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes


input_size = 416

original_image = cv2.imread("kite.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed input        

from matplotlib.pyplot import imshow
imshow(np.asarray(original_image))

import onnxruntime as rt

sess = rt.InferenceSession("yolov4.onnx")

output_name = sess.get_outputs()[0].name
input_name = sess.get_inputs()[0].name

detections = sess.run([output_name], {input_name: image_data})[0]
print("Output shape:", detections.shape)

from config import cfg
import utils as utils
from PIL import Image

ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
XYSCALE = cfg.YOLO.XYSCALE

pred_bbox = utils.postprocess_bbbox(np.expand_dims(detections, axis=0), ANCHORS, STRIDES, XYSCALE)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.213, method='nms')
image = utils.draw_bbox(original_image, bboxes)

image = Image.fromarray(image)
image.show()

imshow(np.asarray(image))





