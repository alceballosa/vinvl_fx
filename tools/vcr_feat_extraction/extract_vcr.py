"""
Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
"""
import argparse
import json
import os.path as op
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import (
    config_dataset_file,
    load_labelmap_file,
)
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from tools.vcr_feat_extraction.detect_utils import detect_objects_on_single_image
from tools.vcr_feat_extraction.visual_utils import draw_bb, draw_rel


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        "white",
        "black",
        "blue",
        "green",
        "red",
        "brown",
        "yellow",
        "small",
        "large",
        "silver",
        "wooden",
        "wood",
        "orange",
        "gray",
        "grey",
        "metal",
        "pink",
        "tall",
        "long",
        "dark",
        "purple",
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {"blonde": "blond"}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]


def save_fx_hdf5(prediction: BoxList, output_file: Path):
    """
    Saves prediction into an H5 file. Note that the classes and attributes
    are saved as class ids, so they must be converted to strings after being
    loaded.

    Parameters:
        prediction (BoxList):
            prediction to save
        output_file (Path):
            path where the prediction will be saved
    """
    image_features = prediction.get_field("box_features")[-1:].numpy()
    box_coords = prediction.bbox.numpy()[:-1]
    box_features = prediction.get_field("box_features")[:-1].numpy()
    predicted_classes = prediction.get_field("labels")[:-1].numpy()
    predicted_scores = prediction.get_field("scores")[:-1].numpy()
    predicted_attr_labels = prediction.get_field("attr_labels")[:-1].numpy()
    predicted_attr_scores = prediction.get_field("attr_scores")[:-1].numpy()

    with h5py.File(str(output_file), "w") as file:
        _ = file.create_dataset("image_features", data=image_features)
        _ = file.create_dataset("box_coords", data=box_coords)
        _ = file.create_dataset("box_features", data=box_features)
        _ = file.create_dataset("predicted_classes", data=predicted_classes)
        _ = file.create_dataset("predicted_class_scores", data=predicted_scores)
        _ = file.create_dataset("predicted_attr_labels", data=predicted_attr_labels)
        _ = file.create_dataset("predicted_attr_scores", data=predicted_attr_scores)


def save_fx_image(
    prediction: BoxList,
    img: np.ndarray,
    path_img: Path,
    output_dir: Path,
    dataset_labelmap,
    dataset_attr_labelmap,
):
    """
    Saves a prediction and the top attributes for each bounding box
    into an image for review.
    """
    # for all lists, remove the last element which corresponds
    # to the full image features
    boxes = np.array(prediction.bbox.tolist()[:-1])
    classes = prediction.get_field("labels").tolist()[:-1]
    scores = np.array(prediction.get_field("scores").tolist()[:-1])
    attr_scores = np.array(prediction.get_field("attr_scores")[:-1])
    attr_labels = prediction.get_field("attr_labels")[:-1]
    dets = [
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
    for obj in dets:
        obj["class"] = dataset_labelmap[obj["class"]]
    for obj in dets:
        obj["attr"], obj["attr_conf"] = postprocess_attr(
            dataset_attr_labelmap, obj["attr"], obj["attr_conf"]
        )
    rects = [d["rect"] for d in dets]
    scores = [d["conf"] for d in dets]
    attr_labels = [",".join(d["attr"]) for d in dets]
    attr_scores = [d["attr_conf"] for d in dets]
    labels = [attr_label + " " + d["class"] for d, attr_label in zip(dets, attr_labels)]
    draw_bb(img, rects, labels, scores)
    save_path = output_dir / path_img.name
    cv2.imwrite(str(save_path), img)
    print(f"Save visualization of results to: {save_path}")
    result_str = ""
    for label, score, attr_score in zip(labels, scores, attr_scores):
        result_str += label + "\n"
        result_str += ",".join([str(conf) for conf in attr_score])
        result_str += "\t" + str(score) + "\n"
    text_save_file = op.splitext(save_path)[0] + ".txt"
    with open(text_save_file, "w", encoding="utf-8") as fid:
        fid.write(result_str)


def process_vcr_image(
    model,
    path_img: Path,
    output_dir: Path,
    cfg,
    dataset_labelmap: dict,
    dataset_attr_labelmap: dict,
):
    """
    Extract ResNext152 features for a single VCR image.
    """

    movie_dir = path_img.parent.name
    output_movie_dir = output_dir / movie_dir
    output_movie_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_movie_dir / path_img.name.replace("jpg", "h5")
    if output_file.exists():
        print(f"Skipping {path_img}, features were calculated already")
        return

    path_metadata = Path(str(path_img).replace("jpg", "json"))
    img = cv2.imread(str(path_img))
    with open(path_metadata, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    boxes = [box[0:4] for box in metadata["boxes"]]
    # add a bounding box for the entire image
    boxes += [[0, 0, img.shape[1] - 1, img.shape[0] - 1]]
    box_list = BoxList(bbox=boxes, image_size=(img.shape[1], img.shape[0]), mode="xyxy")
    # box_list.add_field("labels", metadata["names"])
    transforms = build_transforms(cfg, is_train=False)
    prediction = detect_objects_on_single_image(model, transforms, img, box_list)
    save_fx_hdf5(prediction, output_file)
    save_fx_image(
        prediction,
        img,
        path_img,
        output_movie_dir,
        dataset_labelmap,
        dataset_attr_labelmap,
    )


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--config_file", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--path_dataset",
        metavar="FILE",
        help="path to the folder containing the VCR movie folders",
    )
    parser.add_argument(
        "--labelmap_file",
        metavar="FILE",
        help="labelmap file to select classes for visualizatioin",
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        type=str,
        default=None,
        help="dir to save the extracted features",
    )
    parser.add_argument(
        "--visualize_attr", action="store_true", help="visualize the object attributes"
    )
    parser.add_argument(
        "--visualize_relation", action="store_true", help="visualize the relationships"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    args = parser.parse_args()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES = True

    cfg.freeze()

    assert op.isdir(
        args.path_dataset
    ), f"Dataset folder: {args.path_dataset} does not exist"

    # read and process all lines in the jsonl file
    path_vcr_dataset = Path(args.path_dataset)
    list_images = []
    for path_movie in path_vcr_dataset.glob("*"):
        list_images.extend(path_movie.glob("*.jpg"))

    output_dir = cfg.OUTPUT_DIR

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    # dataset labelmap is used to convert the prediction to class labels
    dataset_labelmap_file = config_dataset_file(
        cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE
    )
    assert dataset_labelmap_file
    dataset_allmap = json.load(open(dataset_labelmap_file, "r", encoding="utf-8"))
    dataset_labelmap = {
        int(val): key for key, val in dataset_allmap["label_to_idx"].items()
    }

    dataset_attr_labelmap = {
        int(val): key for key, val in dataset_allmap["attribute_to_idx"].items()
    }

    # for idx in tqdm(range(len(items))):
    # TODO: reimplement for loop
    n_images = len(list_images)
    for idx in tqdm(range(n_images)):
        process_vcr_image(
            model,
            list_images[idx],
            Path(args.save_dir),
            cfg,
            dataset_labelmap,
            dataset_attr_labelmap,
        )


if __name__ == "__main__":
    main()
