import copy
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import SharedArray
import torch.utils.data as torch_data

from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..argo2.argo2_dataset import Argo2Dataset
from ..custom.custom_dataset import CustomDataset
from ..dataset import DatasetTemplate
from ..kitti import kitti_utils
from ..kitti.kitti_dataset import KittiDataset
from ..nuscenes.nuscenes_dataset import NuScenesDataset
from ..once.once_dataset import ONCEDataset
from ..processor.intra_domain_point_mixup import (
    intra_domain_point_mixup,
    intra_domain_point_mixup_cd,
)
from ..waymo.waymo_dataset import WaymoDataset


class AllinOneDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
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
            dataset_cfg=dataset_cfg.KITTI,
            class_names=class_names['kitti'],
            training=training,
            root_path=root_path,
            logger=logger,
        )

        self.kitti_dataset = KittiDataset(
            dataset_cfg=dataset_cfg.KITTI,
            class_names=class_names['kitti'],
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.waymo_dataset = WaymoDataset(
            dataset_cfg=dataset_cfg.WAYMO,
            class_names=class_names['waymo'],
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg.NUSCENES,
            class_names=class_names['nuscenes'],
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.nia_dataset = CustomDataset(
            dataset_cfg=dataset_cfg.NIA,
            class_names=class_names['nia'],
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.argo2_dataset = Argo2Dataset(
            dataset_cfg=dataset_cfg.ARGO2,
            class_names=class_names['argo2'],
            training=training,
            root_path=root_path,
            logger=logger,
        )

        self.once_dataset = ONCEDataset(
            dataset_cfg=dataset_cfg.ONCE,
            class_names=class_names['once'],
            training=training,
            root_path=root_path,
            logger=logger,
        )

        self.class_names = class_names
        self.training = training
        self.kitti_dataset_cfg = dataset_cfg.KITTI  # TODO
        self.waymo_dataset_cfg = dataset_cfg.WAYMO
        self.nuscenes_dataset_cfg = dataset_cfg.NUSCENES
        self.nia_dataset_cfg = dataset_cfg.NIA
        self.argo2_dataset_cfg = dataset_cfg.ARGO2
        self.once_dataset_cfg = dataset_cfg.ONCE

        self.__all__ = [
            self.kitti_dataset,
            self.waymo_dataset,
            self.nuscenes_dataset,
            self.nia_dataset,
            self.argo2_dataset,
            self.once_dataset,
        ]

        self.kitti_data_cnt = len(self.kitti_dataset.kitti_infos)
        self.waymo_data_cnt = len(self.waymo_dataset.infos)
        self.nuscenes_data_cnt = len(self.nuscenes_dataset.infos)
        self.nia_dataset_cnt = len(self.nia_dataset.custom_infos)
        self.argo2_dataset_cnt = len(self.argo2_dataset.argo2_infos)
        self.once_dataset_cnt = len(self.once_dataset.once_infos)

    def nuscenes_getitem(self, index):
        if len(
            self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST
        ) == 0 or np.random.random(
            1
        ) > self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.get(
            'PROB', 0
        ):
            info = copy.deepcopy(self.nuscenes_dataset.infos[index])
            points = self.nuscenes_dataset.get_lidar_with_sweeps(
                index, max_sweeps=self.nuscenes_dataset_cfg.MAX_SWEEPS
            )

            input_dict = {
                'points': points,
                'frame_id': Path(info['lidar_path']).stem,
                'metadata': {'token': info['token']},
            }

            if 'gt_boxes' in info:
                if self.nuscenes_dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (
                        info['num_lidar_pts']
                        > self.nuscenes_dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1
                    )
                else:
                    mask = None
                ## CONVERT CLASS like flustum class
                for k in range(info['gt_names'].shape[0]):
                    info['gt_names'][k] = (
                        self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS[
                            info['gt_names'][k]
                        ]
                        if info['gt_names'][k]
                        in self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS.keys()
                        else info['gt_names'][k]
                    )
                # print(info['gt_names'])
                input_dict.update(
                    {
                        'gt_names': info['gt_names']
                        if mask is None
                        else info['gt_names'][mask],
                        'gt_boxes': info['gt_boxes']
                        if mask is None
                        else info['gt_boxes'][mask],
                    }
                )
                # if self.nuscenes_dataset_cfg.get('SHIFT_COOR', None):
                #     input_dict['gt_boxes'][
                #         :, 0:3
                #     ] += self.nuscenes_dataset_cfg.SHIFT_COOR

            data_dict = self.nuscenes_dataset.prepare_data(data_dict=input_dict)

            if (
                self.nuscenes_dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False)
                and 'gt_boxes' in info
            ):
                gt_boxes = data_dict['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict['gt_boxes'] = gt_boxes

            if not self.nuscenes_dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
                data_dict['gt_boxes'] = data_dict['gt_boxes'][
                    :, [0, 1, 2, 3, 4, 5, 6, -1]
                ]
        else:
            # idx1 = np.random.randint(len(self.nuscenes_dataset.gt_infos))
            idx2 = np.random.randint(len(self.nuscenes_dataset.infos))
            nus_info_1 = copy.deepcopy(self.nuscenes_dataset.infos[index])
            nus_info_2 = copy.deepcopy(self.nuscenes_dataset.infos[idx2])
            nus_points_1 = self.nuscenes_dataset.get_gt_lidar_with_sweeps(
                index, max_sweeps=self.nuscenes_dataset_cfg.MAX_SWEEPS
            )
            nus_points_2 = self.nuscenes_dataset.get_gt_lidar_with_sweeps(
                idx2, max_sweeps=self.nuscenes_dataset_cfg.MAX_SWEEPS
            )
            # TODO : prob for appling shift
            # if self.nuscenes_dataset_cfg.get('SHIFT_COOR', None):
            #     nus_points_1[:, 0:3] += np.array(
            #         self.nuscenes_dataset_cfg.SHIFT_COOR, dtype=np.float32
            #     )
            #     nus_points_2[:, 0:3] += np.array(
            #         self.nuscenes_dataset_cfg.SHIFT_COOR, dtype=np.float32
            #     )

            input_dict_1 = {
                'points': nus_points_1,
                'frame_id': Path(nus_info_1['lidar_path']).stem,
                'metadata': {'token': nus_info_1['token']},
            }

            input_dict_2 = {
                'points': nus_points_2,
                'frame_id': Path(nus_info_2['lidar_path']).stem,
                'metadata': {'token': nus_info_2['token']},
            }
            assert 'gt_boxes' in nus_info_1
            assert 'gt_boxes' in nus_info_2

            if self.nuscenes_dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask_1 = (
                    nus_info_1['num_lidar_pts']
                    > self.nuscenes_dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1
                )
                mask_2 = (
                    nus_info_2['num_lidar_pts']
                    > self.nuscenes_dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1
                )
            else:
                mask_1 = None
                mask_2 = None

            for k in range(nus_info_1['gt_names'].shape[0]):
                nus_info_1['gt_names'][k] = (
                    self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS[
                        nus_info_1['gt_names'][k]
                    ]
                    if nus_info_1['gt_names'][k]
                    in self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS.keys()
                    else nus_info_1['gt_names'][k]
                )

            for k in range(nus_info_2['gt_names'].shape[0]):
                nus_info_2['gt_names'][k] = (
                    self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS[
                        nus_info_2['gt_names'][k]
                    ]
                    if nus_info_2['gt_names'][k]
                    in self.nuscenes_dataset_cfg.MAP_NUSCENES_TO_CLASS.keys()
                    else nus_info_2['gt_names'][k]
                )

            input_dict_1.update(
                {
                    'gt_names': nus_info_1['gt_names']
                    if mask_1 is None
                    else nus_info_1['gt_names'][mask_1],
                    'gt_boxes': nus_info_1['gt_boxes']
                    if mask_1 is None
                    else nus_info_1['gt_boxes'][mask_1],
                }
            )

            input_dict_2.update(
                {
                    'gt_names': nus_info_2['gt_names']
                    if mask_2 is None
                    else nus_info_2['gt_names'][mask_2],
                    'gt_boxes': nus_info_2['gt_boxes']
                    if mask_2 is None
                    else nus_info_2['gt_boxes'][mask_2],
                }
            )

            if not self.nuscenes_dataset_cfg.PRED_VELOCITY:
                input_dict_1['gt_boxes'] = input_dict_1['gt_boxes'][
                    :, [0, 1, 2, 3, 4, 5, 6]
                ]
                input_dict_2['gt_boxes'] = input_dict_2['gt_boxes'][
                    :, [0, 1, 2, 3, 4, 5, 6]
                ]

            # if self.nuscenes_dataset_cfg.get('SHIFT_COOR', None):
            #     input_dict_1['gt_boxes'][:, 0:3] += self.nuscenes_dataset_cfg.SHIFT_COOR
            #     input_dict_2['gt_boxes'][:, 0:3] += self.nuscenes_dataset_cfg.SHIFT_COOR

            if len(self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 1:
                if (
                    self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0]
                    == 'mix_up'
                ):
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if (
                    self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0]
                    == 'cut_mix'
                ):
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)
            else:
                index = np.random.randint(
                    len(self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST)
                )
                if (
                    self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'mix_up'
                ):
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if (
                    self.nuscenes_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'cut_mix'
                ):
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)

            # data_dict = self.prepare_mixup_data(
            #     input_dict_1, input_dict_2, self.class_names['nuscenes']
            # )

            if self.nuscenes_dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
                gt_boxes = data_dict['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict['gt_boxes'] = gt_boxes

        return data_dict

    def nia_getitem(self, index):
        if len(
            self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST
        ) == 0 or np.random.random(1) > self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.get(
            'PROB', 0
        ):
            info = copy.deepcopy(self.nia_dataset.custom_infos[index])
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.nia_dataset.get_lidar(sample_idx)
            input_dict = {
                'frame_id': self.nia_dataset.sample_id_list[index],
                'points': points,
            }

            if 'annos' in info:
                # TODO : mapping cls to kitti
                annos = info['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.nia_dataset_cfg.MAP_NIA_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.nia_dataset_cfg.MAP_NIA_TO_CLASS.keys()
                        else annos['name'][k]
                    )
                gt_names = annos['name']
                gt_boxes_lidar = annos['gt_boxes_lidar']
                input_dict.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})
            data_dict = self.nia_dataset.prepare_data(data_dict=input_dict)

        else:
            idx2 = np.random.randint(len(self.nia_dataset.custom_infos))
            info_1 = copy.deepcopy(self.nia_dataset.custom_infos[index])
            info_2 = copy.deepcopy(self.nia_dataset.custom_infos[idx2])

            # sample_idx = info['point_cloud']['lidar_idx']
            points_1 = self.nia_dataset.get_lidar(info_1['point_cloud']['lidar_idx'])
            points_2 = self.nia_dataset.get_lidar(info_2['point_cloud']['lidar_idx'])

            input_dict_1 = {
                'frame_id': self.nia_dataset.sample_id_list[index],
                'points': points_1,
            }
            input_dict_2 = {
                'frame_id': self.nia_dataset.sample_id_list[idx2],
                'points': points_2,
            }

            if 'annos' in info_1:
                # TODO : mapping cls to kitti
                annos = info_1['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.nia_dataset_cfg.MAP_NIA_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.nia_dataset_cfg.MAP_NIA_TO_CLASS.keys()
                        else annos['name'][k]
                    )
                gt_names = annos['name']
                gt_boxes_lidar = annos['gt_boxes_lidar']
                input_dict_1.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})

            if 'annos' in info_2:
                # TODO : mapping cls to kitti
                annos = info_2['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.nia_dataset_cfg.MAP_NIA_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.nia_dataset_cfg.MAP_NIA_TO_CLASS.keys()
                        else annos['name'][k]
                    )
                gt_names = annos['name']
                gt_boxes_lidar = annos['gt_boxes_lidar']
                input_dict_2.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})

            if len(self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 1:
                if self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'cut_mix':
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)
            else:
                index = np.random.randint(
                    len(self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST)
                )
                if self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if (
                    self.nia_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'cut_mix'
                ):
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)

        return data_dict

    def once_getitem(self, index):
        # self.once_dataset_cfg = dataset_cfg.ONCE
        if len(self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 0 or np.random.random(
            1
        ) > self.dataset_cfg.DATA_AUGMENTOR.MIX.get('PROB', 0):
            info = copy.deepcopy(self.once_dataset.once_infos[index])
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.once_dataset.get_lidar(seq_id, frame_id)

            if self.once_dataset_cfg.get('POINT_PAINTING', False):
                points = self.once_dataset.point_painting(points, info)

            input_dict = {
                'points': points,
                'frame_id': frame_id,
            }

            if 'annos' in info:
                annos = info['annos']

                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.once_dataset_cfg.MAP_ONCE_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.once_dataset_cfg.MAP_ONCE_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                input_dict.update(
                    {
                        'gt_names': annos['name'],
                        'gt_boxes': annos['boxes_3d'],
                        'num_points_in_gt': annos.get('num_points_in_gt', None),
                    }
                )

            data_dict = self.once_dataset.prepare_data(data_dict=input_dict)
            data_dict.pop('num_points_in_gt', None)

        else:
            idx2 = np.random.randint(len(self.once_dataset.once_infos))
            once_info_1 = copy.deepcopy(self.once_dataset.once_infos[index])
            once_info_2 = copy.deepcopy(self.once_dataset.once_infos[idx2])
            once_points_1 = self.once_dataset.get_lidar(
                once_info_1['sequence_id'], once_info_1['frame_id']
            )
            once_points_2 = self.once_dataset.get_lidar(
                once_info_2['sequence_id'], once_info_2['frame_id']
            )

            if self.dataset_cfg.get('POINT_PAINTING', False):
                once_points_1 = self.point_painting(once_points_1, once_info_1)
                once_points_2 = self.point_painting(once_points_2, once_info_2)
            #  frame_id = info['frame_id']
            # seq_id = info['sequence_id']
            input_dict_1 = {
                'points': once_points_1,
                'frame_id': once_info_1['frame_id'],
            }

            input_dict_2 = {
                'points': once_points_2,
                'frame_id': once_info_2['frame_id'],
            }
            if 'annos' in once_info_1:
                annos = once_info_1['annos']

                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.once_dataset_cfg.MAP_ONCE_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.once_dataset_cfg.MAP_ONCE_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                input_dict_1.update(
                    {
                        'gt_names': annos['name'],
                        'gt_boxes': annos['boxes_3d'],
                        'num_points_in_gt': annos.get('num_points_in_gt', None),
                    }
                )

            if 'annos' in once_info_2:
                annos = once_info_2['annos']

                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.once_dataset_cfg.MAP_ONCE_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.once_dataset_cfg.MAP_ONCE_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                input_dict_2.update(
                    {
                        'gt_names': annos['name'],
                        'gt_boxes': annos['boxes_3d'],
                        'num_points_in_gt': annos.get('num_points_in_gt', None),
                    }
                )

            if len(self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 1:
                if self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'cut_mix':
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)
            else:
                index = np.random.randint(
                    len(self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST)
                )
                if self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if self.dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index] == 'cut_mix':
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)

            data_dict.pop('num_points_in_gt', None)

        return data_dict

    def argo2_getitem(self, index):
        if len(
            self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST
        ) == 0 or np.random.random(1) > self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.get(
            'PROB', 0
        ):
            info = copy.deepcopy(self.argo2_dataset.argo2_infos[index])

            sample_idx = (
                info['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin')
            )
            calib = None
            get_item_list = self.argo2_dataset_cfg.get('GET_ITEM_LIST', ['points'])

            input_dict = {
                'frame_id': sample_idx,
                'calib': calib,
            }

            if 'annos' in info:
                annos = info['annos']

                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                    #   for k in range(annos['name'].shape[0]):
                    # annos['name'][k] = (
                    #     map_waymo_to_kitti[annos['name'][k]]
                    #     if annos['name'][k]
                    #     in self.waymo_dataset_cfg.MAP_WAYMO_TO_CLASS.keys()
                    #     else annos['name'][k]
                    # )

                gt_names = annos['name']
                gt_bboxes_3d = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)

                input_dict.update({'gt_names': gt_names, 'gt_boxes': gt_bboxes_3d})

            if "points" in get_item_list:
                points = self.argo2_dataset.get_lidar(sample_idx)
                input_dict['points'] = points

            input_dict['calib'] = calib
            data_dict = self.argo2_dataset.prepare_data(data_dict=input_dict)
        else:
            info_1 = copy.deepcopy(self.argo2_dataset.argo2_infos[index])
            idx2 = np.random.randint(len(self.argo2_dataset.argo2_infos))
            info_2 = copy.deepcopy(self.argo2_dataset.argo2_infos[idx2])

            sample_idx_1 = (
                info_1['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin')
            )
            sample_idx_2 = (
                info_2['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin')
            )
            calib = None
            get_item_list = self.argo2_dataset_cfg.get('GET_ITEM_LIST', ['points'])

            input_dict_1 = {
                'frame_id': sample_idx_1,
                'calib': calib,
            }
            input_dict_2 = {
                'frame_id': sample_idx_2,
                'calib': calib,
            }

            if 'annos' in info_1:
                annos = info_1['annos']

                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                gt_names = annos['name']
                gt_bboxes_3d = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)

                input_dict_1.update({'gt_names': gt_names, 'gt_boxes': gt_bboxes_3d})

            if 'annos' in info_2:
                annos = info_2['annos']

                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.argo2_dataset_cfg.MAP_ARGO2_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                gt_names = annos['name']
                gt_bboxes_3d = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)

                input_dict.update({'gt_names': gt_names, 'gt_boxes': gt_bboxes_3d})

            if "points" in get_item_list:
                points_1 = self.argo2_dataset.get_lidar(sample_idx_1)
                input_dict_1['points'] = points_1
                points_2 = self.argo2_dataset.get_lidar(sample_idx_2)
                input_dict_2['points'] = points_2

            points_1['calib'] = calib
            points_2['calib'] = calib

            if len(self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 1:
                if self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'cut_mix':
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)
            else:
                index = np.random.randint(
                    len(self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST)
                )
                if (
                    self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'mix_up'
                ):
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if (
                    self.argo2_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'cut_mix'
                ):
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)

            # data_dict = self.argo2_dataset.prepare_data(data_dict=input_dict)

        return data_dict

    def waymo_getitem(self, index):
        # print("getitem waymo")
        info = copy.deepcopy(self.waymo_dataset.infos[index])

        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {'sample_idx': sample_idx}
        if (
            self.waymo_dataset.use_shared_memory
            and index < self.waymo_dataset.shared_memory_file_limit
        ):
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.waymo_dataset.get_lidar(sequence_name, sample_idx)

        if (
            self.waymo_dataset_cfg.get('SEQUENCE_CONFIG', None) is not None
            and self.waymo_dataset_cfg.SEQUENCE_CONFIG.ENABLED
        ):
            (
                points,
                num_points_all,
                sample_idx_pre_list,
                poses,
                pred_boxes,
                pred_scores,
                pred_labels,
            ) = self.waymo_dataset.get_sequence_data(
                info,
                points,
                sequence_name,
                sample_idx,
                self.waymo_dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.waymo_dataset_cfg.get('USE_PREDBOX', False),
            )
            input_dict['poses'] = poses
            if self.waymo_dataset_cfg.get('USE_PREDBOX', False):
                input_dict.update(
                    {
                        'roi_boxes': pred_boxes,
                        'roi_scores': pred_scores,
                        'roi_labels': pred_labels,
                    }
                )

        input_dict.update(
            {
                'points': points,
                'frame_id': info['frame_id'],
            }
        )

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            for k in range(annos['name'].shape[0]):
                annos['name'][k] = (
                    self.waymo_dataset_cfg.MAP_WAYMO_TO_CLASS[annos['name'][k]]
                    if annos['name'][k]
                    in self.waymo_dataset_cfg.MAP_WAYMO_TO_CLASS.keys()
                    else annos['name'][k]
                )

                # map_waymo_to_kitti[annos['name'][k]]

            #  for k in range(info['gt_names'].shape[0]):
            #                 info['gt_names'][k] = (
            #                     map_nuscenes_to_kitti[info['gt_names'][k]]
            #                     if info['gt_names'][k] in map_nuscenes_to_kitti.keys()
            #                     else info['gt_names'][k]
            #                 )

            if self.waymo_dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                    annos['gt_boxes_lidar']
                )
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.waymo_dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.waymo_dataset_cfg.get(
                'FILTER_EMPTY_BOXES_FOR_TRAIN', False
            ):
                mask = annos['num_points_in_gt'] > 0  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]
            input_dict.update(
                {
                    'gt_names': annos['name'],
                    'gt_boxes': gt_boxes_lidar,
                    'num_points_in_gt': annos.get('num_points_in_gt', None),
                }
            )

        data_dict = self.waymo_dataset.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def kitti_getitem(self, index):
        if len(
            self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST
        ) == 0 or np.random.random(1) > self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.get(
            'PROB', 0
        ):
            info = copy.deepcopy(self.kitti_dataset.kitti_infos[index])

            sample_idx = info['point_cloud']['lidar_idx']

            img_shape = info['image']['image_shape']
            calib = self.kitti_dataset.get_calib(sample_idx)
            get_item_list = self.kitti_dataset_cfg.get('GET_ITEM_LIST', ['points'])

            input_dict = {
                'frame_id': sample_idx,
                'calib': calib,
            }

            if 'annos' in info:
                annos = info['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                gt_names = annos['name']
                gt_boxes_camera = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, calib
                )

                input_dict.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})
                if "gt_boxes2d" in get_item_list:
                    input_dict['gt_boxes2d'] = annos["bbox"]
            road_plane = self.kitti_dataset.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

            if "points" in get_item_list:
                # print(f'sample idx : {sample_idx}')
                points = self.kitti_dataset.get_lidar(sample_idx)
                if self.kitti_dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.kitti_dataset.get_fov_flag(
                        pts_rect, img_shape, calib
                    )
                    points = points[fov_flag]
                input_dict['points'] = points

            if "images" in get_item_list:
                input_dict['images'] = self.kitti_dataset.get_image(sample_idx)

            if "depth_maps" in get_item_list:
                input_dict['depth_maps'] = self.kitti_dataset.get_depth_map(sample_idx)

            if "calib_matricies" in get_item_list:
                (
                    input_dict["trans_lidar_to_cam"],
                    input_dict["trans_cam_to_img"],
                ) = kitti_utils.calib_to_matricies(calib)

            input_dict['calib'] = calib
            data_dict = self.kitti_dataset.prepare_data(data_dict=input_dict)

            data_dict['image_shape'] = img_shape
        else:
            info_1 = copy.deepcopy(self.kitti_dataset.kitti_infos[index])
            idx2 = np.random.randint(len(self.kitti_dataset.kitti_infos))
            info_2 = copy.deepcopy(self.kitti_dataset.kitti_infos[idx2])

            sample_idx_1 = info_1['point_cloud']['lidar_idx']
            sample_idx_2 = info_2['point_cloud']['lidar_idx']

            img_shape_1 = info_1['image']['image_shape']
            img_shape_2 = info_2['image']['image_shape']

            calib_1 = self.kitti_dataset.get_calib(sample_idx_1)
            calib_2 = self.kitti_dataset.get_calib(sample_idx_2)

            get_item_list = self.kitti_dataset_cfg.get('GET_ITEM_LIST', ['points'])

            input_dict_1 = {
                'frame_id': sample_idx_1,
                'calib': calib_1,
            }
            input_dict_2 = {
                'frame_id': sample_idx_2,
                'calib': calib_2,
            }

            if 'annos' in info_1:
                annos = info_1['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                gt_names = annos['name']
                gt_boxes_camera = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, calib_1
                )

                input_dict_1.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})
                if "gt_boxes2d" in get_item_list:
                    input_dict_1['gt_boxes2d'] = annos["bbox"]

            road_plane = self.kitti_dataset.get_road_plane(sample_idx_1)
            if road_plane is not None:
                input_dict_1['road_plane'] = road_plane

            if 'annos' in info_2:
                annos = info_2['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = (
                    annos['location'],
                    annos['dimensions'],
                    annos['rotation_y'],
                )
                for k in range(annos['name'].shape[0]):
                    annos['name'][k] = (
                        self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS[annos['name'][k]]
                        if annos['name'][k]
                        in self.kitti_dataset_cfg.MAP_KITTI_TO_CLASS.keys()
                        else annos['name'][k]
                    )

                gt_names = annos['name']
                gt_boxes_camera = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1
                ).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, calib_2
                )

                input_dict_2.update({'gt_names': gt_names, 'gt_boxes': gt_boxes_lidar})
                if "gt_boxes2d" in get_item_list:
                    input_dict_2['gt_boxes2d'] = annos["bbox"]

            road_plane = self.kitti_dataset.get_road_plane(sample_idx_2)
            if road_plane is not None:
                input_dict_2['road_plane'] = road_plane

            #####

            if "points" in get_item_list:
                # print(f'sample idx : {sample_idx}')
                points_1 = self.kitti_dataset.get_lidar(sample_idx_1)
                if self.kitti_dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_1.lidar_to_rect(points_1[:, 0:3])
                    fov_flag = self.kitti_dataset.get_fov_flag(
                        pts_rect, img_shape, calib_1
                    )
                    points_1 = points_1[fov_flag]
                input_dict_1['points'] = points_1

                points_2 = self.kitti_dataset.get_lidar(sample_idx_2)
                if self.kitti_dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_2.lidar_to_rect(points_2[:, 0:3])
                    fov_flag = self.kitti_dataset.get_fov_flag(
                        pts_rect, img_shape, calib_2
                    )
                    points_2 = points_2[fov_flag]
                input_dict_2['points'] = points_2

            if "images" in get_item_list:
                input_dict_1['images'] = self.kitti_dataset.get_image(sample_idx_1)
                input_dict_2['images'] = self.kitti_dataset.get_image(sample_idx_2)

            if "depth_maps" in get_item_list:
                input_dict_1['depth_maps'] = self.kitti_dataset.get_depth_map(
                    sample_idx_1
                )
                input_dict_2['depth_maps'] = self.kitti_dataset.get_depth_map(
                    sample_idx_2
                )

            if "calib_matricies" in get_item_list:
                (
                    input_dict_1["trans_lidar_to_cam"],
                    input_dict_1["trans_cam_to_img"],
                ) = kitti_utils.calib_to_matricies(calib_1)
                (
                    input_dict_2["trans_lidar_to_cam"],
                    input_dict_2["trans_cam_to_img"],
                ) = kitti_utils.calib_to_matricies(calib_2)

            input_dict_1['calib'] = calib_1
            input_dict_2['calib'] = calib_2
            # data_dict = self.kitti_dataset.prepare_data(data_dict=input_dict)
            # data_dict['image_shape'] = img_shape

            if len(self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST) == 1:
                if self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'mix_up':
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[0] == 'cut_mix':
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)
            else:
                index = np.random.randint(
                    len(self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST)
                )
                if (
                    self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'mix_up'
                ):
                    data_dict = self.prepare_mixup_data(input_dict_1, input_dict_2)

                if (
                    self.kitti_dataset_cfg.DATA_AUGMENTOR.MIX.NAME_LIST[index]
                    == 'cut_mix'
                ):
                    data_dict = self.prepare_cutmix_data(input_dict_1, input_dict_2)

        return data_dict

    def __len__(self):
        return (
            self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
            + self.argo2_dataset_cnt
            + self.once_dataset_cnt
        )

    def __getitem__(self, index):
        if (
            index
            >= self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
            + self.argo2_dataset_cnt
        ):
            once_index = (
                index
                - self.kitti_data_cnt
                - self.waymo_data_cnt
                - self.nuscenes_data_cnt
                - self.nia_dataset_cnt
                - self.argo2_dataset_cnt
            )
            data_dict = self.once_getitem(once_index)
        elif (
            index
            >= self.kitti_data_cnt
            + self.waymo_data_cnt
            + self.nuscenes_data_cnt
            + self.nia_dataset_cnt
        ):
            argo2_index = (
                index
                - self.kitti_data_cnt
                - self.waymo_data_cnt
                - self.nuscenes_data_cnt
                - self.nia_dataset_cnt
            )
            data_dict = self.argo2_getitem(argo2_index)
        elif (
            index >= self.kitti_data_cnt + self.waymo_data_cnt + self.nuscenes_data_cnt
        ):
            nia_index = (
                index
                - self.kitti_data_cnt
                - self.waymo_data_cnt
                - self.nuscenes_data_cnt
            )
            data_dict = self.nia_getitem(nia_index)
        elif index >= self.kitti_data_cnt + self.waymo_data_cnt:
            nuscenes_index = index - self.kitti_data_cnt - self.waymo_data_cnt
            data_dict = self.nuscenes_getitem(nuscenes_index)
        elif index >= self.kitti_data_cnt:
            waymo_index = index - self.kitti_data_cnt
            data_dict = self.waymo_getitem(waymo_index)
        else:
            data_dict = self.kitti_getitem(index)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(
                            coor, ((0, 0), (1, 0)), mode='constant', constant_values=i
                        )
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt, val[0].shape[-1]),
                        dtype=np.float32,
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, : val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, val[0].shape[0], max_gt), dtype=np.float32
                    )
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :, : val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros(
                        (batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32
                    )
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, : val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(
                            desired_size=max_h, cur_size=image.shape[0]
                        )
                        pad_w = common_utils.get_pad_params(
                            desired_size=max_w, cur_size=image.shape[1]
                        )
                        pad_width = (pad_h, pad_w)
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(
                            image,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value,
                        )

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len - len(_points)), (0, 0))
                        points_pad = np.pad(
                            _points,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value,
                        )
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['object_scale_noise', 'object_rotate_noise']:
                    max_noise = max([len(x) for x in val])
                    batch_noise = np.zeros((batch_size, max_noise), dtype=np.float32)
                    for k in range(batch_size):
                        batch_noise[k, : val[k].__len__()] = val[k]
                    ret[key] = batch_noise
                elif key in ['beam_labels']:
                    continue
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
