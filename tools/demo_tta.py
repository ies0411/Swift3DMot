import argparse
import glob
import time
from pathlib import Path

import torch

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import argparse
import os
import sys

# from ..pcdet.utils.nms_3d import nms
import time

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tracking_modules.io import (
    get_frame_det,
    get_saving_dir,
    load_detection,
    save_affinity,
    save_results,
)
from tracking_modules.model import AB3DMOT
from tracking_modules.utils import Config, createFolder, get_subfolder_seq

# (class score, x, y, z, dx,dy,dz)


def iou(box_a, box_b):
    box_a_top_right_corner = [box_a[1] + box_a[4], box_a[2] + box_a[5]]
    box_b_top_right_corner = [box_b[1] + box_b[4], box_b[2] + box_b[5]]

    box_a_area = (box_a[4]) * (box_a[5])
    box_b_area = (box_b[4]) * (box_b[5])

    xi = max(box_a[1], box_b[1])
    yi = max(box_a[2], box_b[2])

    corner_x_i = min(box_a_top_right_corner[0], box_b_top_right_corner[0])
    corner_y_i = min(box_a_top_right_corner[1], box_b_top_right_corner[1])

    intersection_area = max(0, corner_x_i - xi) * max(0, corner_y_i - yi)

    intersection_l_min = max(box_a[3], box_b[3])
    intersection_l_max = min(box_a[3] + box_a[6], box_b[3] + box_b[6])
    intersection_length = intersection_l_max - intersection_l_min

    iou = (intersection_area * intersection_length) / float(
        box_a_area * box_a[6]
        + box_b_area * box_b[6]
        - intersection_area * intersection_length
        + 1e-5
    )

    return iou


def nms(original_boxes, iou_threshold=0.5):
    boxes_probability_sorted = original_boxes[np.flip(np.argsort(original_boxes[:, 0]))]
    box_indices = np.arange(0, len(boxes_probability_sorted))
    suppressed_box_indices = []
    tmp_suppress = []

    while len(box_indices) > 0:
        if box_indices[0] not in suppressed_box_indices:
            selected_box = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_box:
                    selected_iou = iou(
                        boxes_probability_sorted[selected_box],
                        boxes_probability_sorted[box_indices[i]],
                    )
                    if selected_iou > iou_threshold:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        box_indices = box_indices[1:]

    preserved_boxes = np.delete(
        boxes_probability_sorted, suppressed_box_indices, axis=0
    )
    return preserved_boxes, suppressed_box_indices


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/pv_rcnn.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/pv_rcnn_8369.pth",
        type=str,
        help="specify the pretrained model",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default="./tracking_modules/configs/config.yml",
        help="tracking config file path",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="confidence threshold of detection",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    tracking_cfg = Config(args.tracking_config)
    return args, cfg, tracking_cfg


# https://gist.github.com/sidharthrajaram/7ba1a06c0bf18c42016cede00381484b
# box_0 = np.array([0.96, 10, 10, 10, 10, 10, 10])  # should make it
# box_1 = np.array([0.90, 10, 10, 10, 11, 11, 12])

# box_2 = np.array([0.76, 21, 10, 13, 10, 9.5, 7])
# box_3 = np.array([0.80, 20.5, 12, 10, 11, 11, 12])
# box_4 = np.array([0.92, 21.5, 11, 10, 10, 10.3, 10])  # should make it

# box_5 = np.array([0.77, 3.9, 2, 2.5, 4, 6.5, 12])
# box_6 = np.array([0.84, 4, 2, 2.5, 4, 6.6, 10])  # should make it
# box_7 = np.array([0.95, 2.99, 2.65, 4.5, 4, 6.35, 12])  # should make it

# box_8 = np.array([0.84, 32, 33, 69, 33.2, 10.2, 6.5])  # should make it

# box_9 = np.array([0.89, 43, 44, 55.5, 11, 11, 12])
# box_10 = np.array([0.93, 41.4, 46, 56.6, 12, 10, 10])  # should make it


# boxes = np.array([box_0, box_1, box_2, box_3,
#                   box_4, box_5, box_6,
#                   box_7, box_8, box_9, box_10])
#   p, s = nms(boxes)
def nms_3d_box(preds, scores, objs, merged_class_map, iou_thres):
    result_bbox = {}
    boxes = []

    if len(preds) == 0:
        print("empty results")
        for class_map in merged_class_map:
            for model_class, suite_class in class_map.items():
                if suite_class == "nothing":
                    continue
                result_bbox[suite_class] = []

        return result_bbox
    for idx, pred in enumerate(preds):
        pred_np = np.array(pred)
        boxes.append(np.append(np.array(scores[idx]), pred_np[:6]))
        # TODO : considering rotation
    _, suppressed_indices = nms(np.array(boxes), iou_thres)
    nms_preds = [x for i, x in enumerate(preds) if i not in suppressed_indices]
    nms_objs = [x for i, x in enumerate(objs) if i not in suppressed_indices]
    for class_map in merged_class_map:
        for model_class, suite_class in class_map.items():
            if suite_class == "nothing":
                continue
            result_bbox[suite_class] = []
        for idx, pred in enumerate(nms_preds):
            label = nms_objs[idx]
            if class_map.get(label, None) is None or class_map[label] == "nothing":
                continue
            matched_suite_class = class_map[label]
            result_bbox[matched_suite_class].append(pred)
    return result_bbox


class TTADataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        tta_type=None,
        ext=".bin",
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list
        # print(dataset_cfg)
        self.data_augmentor = DataAugmentor(
            self.root_path,
            dataset_cfg.TTA.TTA_AUGMENTOR,
            self.class_names,
            logger=self.logger,
        )
        self.tta_queue = []
        # print(dataset_cfg.TTA.TTA_AUGMENTOR)
        if dataset_cfg.TTA.APPLY:
            aug_config_list = (
                dataset_cfg.TTA.TTA_AUGMENTOR
                # if isinstance(self.dataset_cfg.TTA.AUGMENTOR, list)
                # else self.dataset_cfg.TTA.AUGMENTOR.AUG_CONFIG_LIST
            )
            # print(aug_config_list)
            for cur_cfg in aug_config_list:
                # print(cur_cfg)
                # if not isinstance(self.dataset_cfg.TTA.AUGMENTOR, list):
                # print(cur_cfg.)
                # print(cur_cfg.NAME)
                if tta_type == cur_cfg.NAME:
                    # random_beam_downsample
                    # random_beam_upsample
                    cur_augmentor = getattr(self.data_augmentor, cur_cfg.NAME)(
                        config=cur_cfg
                    )
                    self.tta_queue.append(cur_augmentor)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {"points": points, "frame_id": index}

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.TTA.APPLY:
            # for cur_augmentor in self.data_augmentor.data_augmentor_queue:
            for cur_augmentor in self.tta_queue:
                data_dict = cur_augmentor(data_dict=data_dict)
            # for cur_tta in self.tta_queue:
            #     data_dict = cur_tta(data_dict=data_dict)
        return data_dict


def apply_seperate(pred_dicts, tracking_info_data, num_label, thres):
    bbox = {}
    score = {}
    for idx in range(num_label):
        bbox[str(idx + 1)] = []
        score[str(idx + 1)] = []

    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        if thres > pred_dicts[0]["pred_scores"][idx]:
            continue

        label = str(pred_dicts[0]["pred_labels"][idx].item())
        bbox[label].append(pred_bbox.tolist())
        score[label].append(pred_dicts[0]["pred_scores"][idx].tolist())

    for idx in range(num_label):
        tracking_info_data["bbox"][str(idx + 1)].append(bbox[str(idx + 1)])
        tracking_info_data["score"][str(idx + 1)].append(score[str(idx + 1)])


def main():
    args, detection_cfg, tracking_cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Superb 3D CAL-------------------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    total_dataset = []

    default_dataset = TTADataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    total_dataset.append(default_dataset)

    downsampling_dataset = TTADataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        tta_type="random_beam_downsample",
        logger=logger,
    )
    total_dataset.append(downsampling_dataset)

    upsampling_dataset = TTADataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        tta_type="random_beam_upsample",
        logger=logger,
    )
    total_dataset.append(upsampling_dataset)
    total_preds = []
    for dataset in total_dataset:
        logger.info(f"Total number of samples: \t{len(dataset)}")

        model = build_network(
            model_cfg=detection_cfg.MODEL,
            num_class=len(detection_cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        tracking_info_data = {}
        tracking_info_data["pcd"] = []
        tracking_info_data["bbox"] = {}
        tracking_info_data["score"] = {}

        for class_idx in range(len(detection_cfg.CLASS_NAMES)):
            tracking_info_data["bbox"][str(class_idx + 1)] = []
            tracking_info_data["score"][str(class_idx + 1)] = []

        # TODO : set batch size
        total_time = time.time()
        inference_time = time.time()
        with torch.no_grad():
            for idx, data_dict in enumerate(dataset):
                data_dict = dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                # print(pred_dicts)
                # print("----")
                tracking_info_data["pcd"].append(data_dict["points"][:, 1:])
                apply_seperate(
                    pred_dicts,
                    tracking_info_data,
                    len(detection_cfg.CLASS_NAMES),
                    args.confidence,
                )
        # print(tracking_info_data['bbox'])
        # print("====")
        total_preds.append(tracking_info_data)

    nms_total_preds = {}

    for preds in total_preds:
        for label_id, pred in preds["bbox"].items():
            nms_total_preds[label_id] = (
                []
                if nms_total_preds.get(label_id) is None
                else nms_total_preds[label_id]
            )
            for idx, bbox in enumerate(pred[0]):
                # print(f'bbox : {bbox}')
                # print(f'score : {preds["score"][frame_id][0][idx]}')
                mns_bbox_list = []

                mns_bbox_list.append(preds["score"][label_id][0][idx])
                mns_bbox_list.extend(bbox[:-1])
                mns_bbox_list = np.array(mns_bbox_list)

                nms_total_preds[label_id].append(mns_bbox_list)

                # mns_bbox_list.append(preds["score"][0])
                # print(f'pred : {pred}')
                # print(f'score : {preds["score"][frame_id]}')
                # bbox_list.append(preds["score"][frame_id])
                # print(f'pred_labels : {preds["pred_labels"]}')
                # print("=====")
            # print("=====")
    print(nms_total_preds)
    for label_id, nms_box in nms_total_preds.items():
        # print(nms_box)
        if len(nms_box) == 0:
            continue
        p, s = nms(np.array(nms_box))
        print(f'p : {p}')
        print(f's : {s}')
        print("=====")
    # nms_total_preds = np.array(nms_total_preds)
    # p, s = nms(nms_total_preds)
    # print(f'nms_total_preds : {nms_total_preds}')
    # print(f'p : {p}')
    # print(f's : {s}')
    # print(nms_total_preds)
    # print(type(nms_total_preds))

    logger.info(f"detection inference time : {time.time()-inference_time}")

    # TODO : nms
    # print()
    # box_10 = np.array([0.93, 41.4, 46, 56.6, 12, 10, 10])  # should make it

    # boxes = np.array([box_0, box_1, box_2, box_3,
    #                   box_4, box_5, box_6,
    #                   box_7, box_8, box_9, box_10])

    tracking_time = time.time()
    tracking_results = []

    ID_start = 1
    tracker = AB3DMOT(tracking_cfg, "Car", ID_init=ID_start)
    for pred_bbox in tracking_info_data["bbox"][str(1)]:
        tracking_result, affi = tracker.track(pred_bbox)
        tracking_result = np.squeeze(tracking_result)
    tracking_results.append(tracking_result.tolist())
    # print(tracking_results)
    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info(f"total time : {time.time()-total_time}")
    logger.info("========= Finish =========")


if __name__ == "__main__":
    main()
