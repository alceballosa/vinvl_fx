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
        predictions = model(img_input, targets=[boxlist_input])
        predictions = [prediction.to(torch.device("cpu")) for prediction in predictions]

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]
    prediction = predictions[0].resize((img_width, img_height))

    dict_output = {}
    # features for the
    dict_output["image_features"] = prediction.get_field("box_features")[-1].numpy()
    dict_output["boxes"] = {}

    # for all lists, remove the last element which corresponds
    # to the full image features
    boxes = np.array(prediction.bbox.tolist()[:-1])
    classes = prediction.get_field("labels").tolist()[:-1]
    scores = np.array(prediction.get_field("scores").tolist()[:-1])
    bbox_features = prediction.get_field("box_features")[:-1].numpy()
    if "attr_scores" in prediction.extra_fields:
        attr_scores = np.array(prediction.get_field("attr_scores")[:-1])
        attr_labels = prediction.get_field("attr_labels")[:-1]
        dict_output["boxes"] = [
            {
                "rect": box,
                "class": cls,
                "feature": feature,
                "conf": score,
                "attr": attr[attr_conf > 0.01].tolist(),
                "attr_conf": attr_conf[attr_conf > 0.01].tolist(),
            }
            for box, cls, feature, score, attr, attr_conf in zip(
                boxes, classes, bbox_features, scores, attr_labels, attr_scores
            )
        ]
        return dict_output

    # TODO: fix this branch
    return [
        {"rect": box, "class": cls, "conf": score}
        for box, cls, score in zip(boxes, classes, scores)
    ]
