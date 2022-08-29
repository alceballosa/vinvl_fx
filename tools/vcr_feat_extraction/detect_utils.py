# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import cv2
import numpy as np
import torch
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from scene_graph_benchmark.AttrRCNN import AttrRCNN


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


# TODO: implement detection on multiple images
def detect_objects_on_single_image(
    model: AttrRCNN, transforms, cv2_img: np.ndarray, boxlist: BoxList
):
    # cv2_img is the original input, so we can get the height and
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)
    boxlist_input = boxlist.resize(size=(img_input.shape[-1], img_input.shape[-2]))

    with torch.no_grad():
        prediction = model(img_input, targets=[boxlist_input])
        prediction = prediction[0].to(torch.device("cpu"))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]
    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field("labels").tolist()
    scores = prediction.get_field("scores").tolist()

    if "attr_scores" in prediction.extra_fields:
        attr_scores = prediction.get_field("attr_scores")
        attr_labels = prediction.get_field("attr_labels")
        return [
            {
                "rect": box,
                "class": cls,
                "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist(),
            }
            for box, cls, score, attr, attr_conf in zip(
                boxes, classes, scores, attr_labels, attr_scores
            )
        ]

    return [
        {"rect": box, "class": cls, "conf": score}
        for box, cls, score in zip(boxes, classes, scores)
    ]
