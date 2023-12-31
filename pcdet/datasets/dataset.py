# from collections import defaultdict
# from pathlib import Path

# import numpy as np
# import torch.utils.data as torch_data

# from ..utils import common_utils
# from .augmentor.data_augmentor import DataAugmentor
# from .processor.data_processor import DataProcessor
# from .processor.point_feature_encoder import PointFeatureEncoder


# class DatasetTemplate(torch_data.Dataset):
#     def __init__(
#         self,
#         dataset_cfg=None,
#         class_names=None,
#         training=True,
#         root_path=None,
#         logger=None,
#     ):
#         super().__init__()
#         self.dataset_cfg = dataset_cfg
#         self.training = training
#         self.class_names = class_names
#         self.logger = logger
#         self.root_path = (
#             root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
#         )
#         self.logger = logger
#         if self.dataset_cfg is None or class_names is None:
#             return

#         self.point_cloud_range = np.array(
#             self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32
#         )
#         self.point_feature_encoder = PointFeatureEncoder(
#             self.dataset_cfg.POINT_FEATURE_ENCODING,
#             point_cloud_range=self.point_cloud_range,
#         )
#         self.data_augmentor = (
#             DataAugmentor(
#                 self.root_path,
#                 self.dataset_cfg.DATA_AUGMENTOR,
#                 self.class_names,
#                 logger=self.logger,
#             )
#             if self.training
#             else None
#         )
#         self.data_processor = DataProcessor(
#             self.dataset_cfg.DATA_PROCESSOR,
#             point_cloud_range=self.point_cloud_range,
#             training=self.training,
#             num_point_features=self.point_feature_encoder.num_point_features,
#         )

#         self.grid_size = self.data_processor.grid_size
#         self.voxel_size = self.data_processor.voxel_size
#         self.total_epochs = 0
#         self._merge_all_iters_to_one_epoch = False

#         if hasattr(self.data_processor, "depth_downsample_factor"):
#             self.depth_downsample_factor = self.data_processor.depth_downsample_factor
#         else:
#             self.depth_downsample_factor = None

#     @property
#     def mode(self):
#         return 'train' if self.training else 'test'

#     def __getstate__(self):
#         d = dict(self.__dict__)
#         del d['logger']
#         return d

#     def __setstate__(self, d):
#         self.__dict__.update(d)

#     def generate_prediction_dicts(
#         self, batch_dict, pred_dicts, class_names, output_path=None
#     ):
#         """
#         Args:
#             batch_dict:
#                 frame_id:
#             pred_dicts: list of pred_dicts
#                 pred_boxes: (N, 7 or 9), Tensor
#                 pred_scores: (N), Tensor
#                 pred_labels: (N), Tensor
#             class_names:
#             output_path:

#         Returns:

#         """

#         def get_template_prediction(num_samples):
#             box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
#             ret_dict = {
#                 'name': np.zeros(num_samples),
#                 'score': np.zeros(num_samples),
#                 'boxes_lidar': np.zeros([num_samples, box_dim]),
#                 'pred_labels': np.zeros(num_samples),
#             }
#             return ret_dict

#         def generate_single_sample_dict(box_dict):
#             pred_scores = box_dict['pred_scores'].cpu().numpy()
#             pred_boxes = box_dict['pred_boxes'].cpu().numpy()
#             pred_labels = box_dict['pred_labels'].cpu().numpy()
#             pred_dict = get_template_prediction(pred_scores.shape[0])
#             if pred_scores.shape[0] == 0:
#                 return pred_dict

#             pred_dict['name'] = np.array(class_names)[pred_labels - 1]
#             pred_dict['score'] = pred_scores
#             pred_dict['boxes_lidar'] = pred_boxes
#             pred_dict['pred_labels'] = pred_labels

#             return pred_dict

#         annos = []
#         for index, box_dict in enumerate(pred_dicts):
#             single_pred_dict = generate_single_sample_dict(box_dict)
#             single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
#             if 'metadata' in batch_dict:
#                 single_pred_dict['metadata'] = batch_dict['metadata'][index]
#             annos.append(single_pred_dict)

#         return annos

#     def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
#         if merge:
#             self._merge_all_iters_to_one_epoch = True
#             self.total_epochs = epochs
#         else:
#             self._merge_all_iters_to_one_epoch = False

#     def __len__(self):
#         raise NotImplementedError

#     def __getitem__(self, index):
#         """
#         To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
#         the unified normative coordinate and call the function self.prepare_data() to process the data and send them
#         to the model.

#         Args:
#             index:

#         Returns:

#         """
#         raise NotImplementedError

#     def prepare_data(self, data_dict):
#         """
#         Args:
#             data_dict:
#                 points: optional, (N, 3 + C_in)
#                 gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
#                 gt_names: optional, (N), string
#                 ...

#         Returns:
#             data_dict:
#                 frame_id: string
#                 points: (N, 3 + C_in)
#                 gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
#                 gt_names: optional, (N), string
#                 use_lead_xyz: bool
#                 voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
#                 voxel_coords: optional (num_voxels, 3)
#                 voxel_num_points: optional (num_voxels)
#                 ...
#         """
#         if self.training:
#             assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
#             gt_boxes_mask = np.array(
#                 [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_
#             )

#             if 'calib' in data_dict:
#                 calib = data_dict['calib']
#             data_dict = self.data_augmentor.forward(
#                 data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
#             )
#             if 'calib' in data_dict:
#                 data_dict['calib'] = calib
#         if data_dict.get('gt_boxes', None) is not None:
#             selected = common_utils.keep_arrays_by_name(
#                 data_dict['gt_names'], self.class_names
#             )
#             data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
#             data_dict['gt_names'] = data_dict['gt_names'][selected]
#             gt_classes = np.array(
#                 [self.class_names.index(n) + 1 for n in data_dict['gt_names']],
#                 dtype=np.int32,
#             )
#             gt_boxes = np.concatenate(
#                 (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
#                 axis=1,
#             )
#             data_dict['gt_boxes'] = gt_boxes

#             if data_dict.get('gt_boxes2d', None) is not None:
#                 data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

#         if data_dict.get('points', None) is not None:
#             data_dict = self.point_feature_encoder.forward(data_dict)

#         data_dict = self.data_processor.forward(data_dict=data_dict)

#         if self.training and len(data_dict['gt_boxes']) == 0:
#             new_index = np.random.randint(self.__len__())
#             return self.__getitem__(new_index)

#         data_dict.pop('gt_names', None)

#         return data_dict

#     @staticmethod
#     def collate_batch(batch_list, _unused=False):
#         data_dict = defaultdict(list)
#         for cur_sample in batch_list:
#             for key, val in cur_sample.items():
#                 data_dict[key].append(val)
#         batch_size = len(batch_list)
#         ret = {}

#         for key, val in data_dict.items():
#             try:
#                 if key in ['voxels', 'voxel_num_points']:
#                     ret[key] = np.concatenate(val, axis=0)
#                 elif key in ['points', 'voxel_coords']:
#                     coors = []
#                     for i, coor in enumerate(val):
#                         coor_pad = np.pad(
#                             coor, ((0, 0), (1, 0)), mode='constant', constant_values=i
#                         )
#                         coors.append(coor_pad)
#                     ret[key] = np.concatenate(coors, axis=0)
#                 elif key in ['gt_boxes']:
#                     max_gt = max([len(x) for x in val])
#                     batch_gt_boxes3d = np.zeros(
#                         (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32
#                     )
#                     for k in range(batch_size):
#                         batch_gt_boxes3d[k, : val[k].__len__(), :] = val[k]
#                     ret[key] = batch_gt_boxes3d

#                 elif key in ['roi_boxes']:
#                     max_gt = max([x.shape[1] for x in val])
#                     batch_gt_boxes3d = np.zeros(
#                         (batch_size, val[0].shape[0], max_gt, val[0].shape[-1]),
#                         dtype=np.float32,
#                     )
#                     for k in range(batch_size):
#                         batch_gt_boxes3d[k, :, : val[k].shape[1], :] = val[k]
#                     ret[key] = batch_gt_boxes3d

#                 elif key in ['roi_scores', 'roi_labels']:
#                     max_gt = max([x.shape[1] for x in val])
#                     batch_gt_boxes3d = np.zeros(
#                         (batch_size, val[0].shape[0], max_gt), dtype=np.float32
#                     )
#                     for k in range(batch_size):
#                         batch_gt_boxes3d[k, :, : val[k].shape[1]] = val[k]
#                     ret[key] = batch_gt_boxes3d

#                 elif key in ['gt_boxes2d']:
#                     max_boxes = 0
#                     max_boxes = max([len(x) for x in val])
#                     batch_boxes2d = np.zeros(
#                         (batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32
#                     )
#                     for k in range(batch_size):
#                         if val[k].size > 0:
#                             batch_boxes2d[k, : val[k].__len__(), :] = val[k]
#                     ret[key] = batch_boxes2d
#                 elif key in ["images", "depth_maps"]:
#                     # Get largest image size (H, W)
#                     max_h = 0
#                     max_w = 0
#                     for image in val:
#                         max_h = max(max_h, image.shape[0])
#                         max_w = max(max_w, image.shape[1])

#                     # Change size of images
#                     images = []
#                     for image in val:
#                         pad_h = common_utils.get_pad_params(
#                             desired_size=max_h, cur_size=image.shape[0]
#                         )
#                         pad_w = common_utils.get_pad_params(
#                             desired_size=max_w, cur_size=image.shape[1]
#                         )
#                         pad_width = (pad_h, pad_w)
#                         pad_value = 0

#                         if key == "images":
#                             pad_width = (pad_h, pad_w, (0, 0))
#                         elif key == "depth_maps":
#                             pad_width = (pad_h, pad_w)

#                         image_pad = np.pad(
#                             image,
#                             pad_width=pad_width,
#                             mode='constant',
#                             constant_values=pad_value,
#                         )

#                         images.append(image_pad)
#                     ret[key] = np.stack(images, axis=0)
#                 elif key in ['calib']:
#                     ret[key] = val
#                 elif key in ["points_2d"]:
#                     max_len = max([len(_val) for _val in val])
#                     pad_value = 0
#                     points = []
#                     for _points in val:
#                         pad_width = ((0, max_len - len(_points)), (0, 0))
#                         points_pad = np.pad(
#                             _points,
#                             pad_width=pad_width,
#                             mode='constant',
#                             constant_values=pad_value,
#                         )
#                         points.append(points_pad)
#                     ret[key] = np.stack(points, axis=0)
#                 else:
#                     ret[key] = np.stack(val, axis=0)
#             except:
#                 print('Error in collate_batch: key=%s' % key)
#                 raise TypeError

#         ret['batch_size'] = batch_size
#         return ret

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.inter_domain_point_cutmix import inter_domain_point_cutmix
from .processor.intra_domain_point_mixup import (
    intra_domain_point_mixup,
    intra_domain_point_mixup_cd,
)
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(
        self,
        dataset_cfg=None,
        class_names=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = (
            root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        )

        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(
            self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32
        )
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range,
        )
        self.data_augmentor = (
            DataAugmentor(
                self.root_path,
                self.dataset_cfg.DATA_AUGMENTOR,
                self.class_names,
                logger=self.logger,
            )
            if self.training
            else None
        )
        if self.dataset_cfg.get("AUGMENT_RANDOMLY"):
            self.augment_randomly = self.dataset_cfg.AUGMENT_RANDOMLY
        else:
            self.augment_randomly = False
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features,
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def prepare_cutmix_data(self, data_dict_source, data_dict_target):
        if self.training:
            assert (
                'gt_boxes' in data_dict_source
            ), 'gt_boxes should be provided for training in data_dict_source!'
            assert (
                'gt_boxes' in data_dict_target
            ), 'gt_boxes should be proviced for training in data_dict_target!'

            gt_boxes_mask_source = np.array(
                [n in self.class_names for n in data_dict_source['gt_names']],
                dtype=np.bool_,
            )
            gt_boxes_mask_target = np.array(
                [n in self.class_names for n in data_dict_target['gt_names']],
                dtype=np.bool_,
            )

            # data_dict_source = self.data_augmentor.forward_randomly(
            #     data_dict={**data_dict_source, 'gt_boxes_mask': gt_boxes_mask_source}
            # )

            # data_dict_target = self.data_augmentor.forward_randomly(
            #     data_dict={**data_dict_target, 'gt_boxes_mask': gt_boxes_mask_target}
            # )

        if data_dict_source.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict_source['gt_names'], self.class_names
            )
            data_dict_source['gt_boxes'] = data_dict_source['gt_boxes'][selected]
            data_dict_source['gt_names'] = data_dict_source['gt_names'][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict_source['gt_names']],
                dtype=np.int32,
            )

            for idx, name in enumerate(data_dict_source['gt_names']):
                if name == self.class_names[0]:
                    data_dict_source['gt_names'][idx] = self.class_names[0]

            gt_boxes = np.concatenate(
                (
                    data_dict_source['gt_boxes'],
                    gt_classes.reshape(-1, 1).astype(np.float32),
                ),
                axis=1,
            )
            data_dict_source['gt_boxes'] = gt_boxes

            if data_dict_source.get('gt_boxes2d', None) is not None:
                data_dict_source['gt_boxes2d'] = data_dict_source['gt_boxes2d'][
                    selected
                ]

        if data_dict_target.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict_target['gt_names'], self.class_names
            )
            data_dict_target['gt_boxes'] = data_dict_target['gt_boxes'][selected]
            data_dict_target['gt_names'] = data_dict_target['gt_names'][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict_target['gt_names']],
                dtype=np.int32,
            )

            for idx, name in enumerate(data_dict_target['gt_names']):
                if name == self.class_names[0]:
                    data_dict_target['gt_names'][idx] = self.class_names[0]

            gt_boxes = np.concatenate(
                (
                    data_dict_target['gt_boxes'],
                    gt_classes.reshape(-1, 1).astype(np.float32),
                ),
                axis=1,
            )
            data_dict_target['gt_boxes'] = gt_boxes

            if data_dict_target.get('gt_boxes2d', None) is not None:
                data_dict_target['gt_boxes2d'] = data_dict_target['gt_boxes2d'][
                    selected
                ]

        if data_dict_source.get('points', None) is not None:
            data_dict_source = self.point_feature_encoder.forward(data_dict_source)

        if data_dict_target.get('points', None) is not None:
            data_dict_target = self.point_feature_encoder.forward(data_dict_target)

        assert data_dict_source is not None and data_dict_target is not None

        # if self.dataset_cfg.MIX_TYPE == 'cutmix':
        # new_data_dict = random_patch_replacement(data_dict_source, data_dict_target, self.point_cloud_range, self.dataset_cfg.CROP_RANGE_PERCENT)
        # new_data_dict = random_patch_replacement(data_dict_source, data_dict_target, self.point_cloud_range)
        cutmixed_data_dict = inter_domain_point_cutmix(
            data_dict_source, data_dict_target, self.point_cloud_range
        )
        # print("to here")
        # new_data_dict_1, new_data_dict_2 = new_data_dict[0], new_data_dict[1]
        # else:
        #     raise NotImplementedError

        # print(cutmixed_data_dict.keys())
        if len(cutmixed_data_dict['gt_boxes'].shape) != 2:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        # if len(new_data_dict_1['gt_boxes'].shape) != 2 or len(new_data_dict_2['gt_boxes'].shape) != 2:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        cutmixed_data_dict = self.data_processor.forward(cutmixed_data_dict)

        # new_data_dict_1 = self.data_processor.forward(data_dict=new_data_dict_1)
        # new_data_dict_2 = self.data_processor.forward(data_dict=new_data_dict_2)

        if self.training and len(cutmixed_data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        # if self.training and (len(new_data_dict_1['gt_boxes']) == 0 or len(new_data_dict_2['gt_boxes']) == 0):
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        cutmixed_data_dict.pop('gt_names', None)
        # new_data_dict_1.pop('gt_names', None)
        # new_data_dict_2.pop('gt_names', None)

        return cutmixed_data_dict

    def prepare_mixup_data(self, data_dict_1, data_dict_2):
        if self.training:
            assert 'gt_boxes' in data_dict_1
            assert 'gt_boxes' in data_dict_2
            gt_boxes_mask_1 = np.array(
                [n in self.class_names for n in data_dict_1['gt_names']], dtype=np.bool_
            )
            gt_boxes_mask_2 = np.array(
                [n in self.class_names for n in data_dict_2['gt_names']], dtype=np.bool_
            )
            if 'calib' in data_dict_1:
                calib_1 = data_dict_1['calib']
            if 'calib' in data_dict_2:
                calib_2 = data_dict_2['calib']
            # data_dict_1 = self.data_augmentor.forward_randomly(
            #     data_dict={**data_dict_1, 'gt_boxes_mask': gt_boxes_mask_1}
            # )

            # data_dict_2 = self.data_augmentor.forward_randomly(
            #     data_dict={**data_dict_2, 'gt_boxes_mask': gt_boxes_mask_2}
            # )

        if data_dict_1.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict_1['gt_names'], self.class_names
            )

            data_dict_1['gt_boxes'] = data_dict_1['gt_boxes'][selected]
            data_dict_1['gt_names'] = data_dict_1['gt_names'][selected]

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict_1['gt_names']],
                dtype=np.int32,
            )
            gt_boxes = np.concatenate(
                (data_dict_1['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
                axis=1,
            )

            data_dict_1['gt_boxes'] = gt_boxes

            if data_dict_1.get('gt_boxes2d', None) is not None:
                data_dict_1['gt_boxes2d'] = data_dict_1['gt_boxes2d'][selected]

        if data_dict_2.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict_2['gt_names'], self.class_names
            )
            data_dict_2['gt_boxes'] = data_dict_2['gt_boxes'][selected]
            data_dict_2['gt_names'] = data_dict_2['gt_names'][selected]

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict_2['gt_names']],
                dtype=np.int32,
            )
            gt_boxes = np.concatenate(
                (data_dict_2['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
                axis=1,
            )

            data_dict_2['gt_boxes'] = gt_boxes

            if data_dict_2.get('gt_boxes2d', None) is not None:
                data_dict_2['gt_boxes2d'] = data_dict_2['gt_boxes2d'][selected]

        if data_dict_1.get('points', None) is not None:
            data_dict_1 = self.point_feature_encoder.forward(data_dict_1)

        if data_dict_2.get('points', None) is not None:
            data_dict_2 = self.point_feature_encoder.forward(data_dict_2)

        if self.dataset_cfg.DATA_AUGMENTOR.get('COLLISION_DETECTION', True):
            data_dict = intra_domain_point_mixup_cd(
                data_dict_1,
                data_dict_2,
                alpha=self.dataset_cfg.DATA_AUGMENTOR.MIX.ALPHA,
            )
        else:
            data_dict = intra_domain_point_mixup(
                data_dict_1,
                data_dict_2,
                alpha=self.dataset_cfg.DATA_AUGMENTOR.MIX.ALPHA,
            )

        if len(data_dict['gt_boxes'].shape) != 2:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(
        self, batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]),
                'pred_labels': np.zeros(num_samples),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_
            )

            if 'calib' in data_dict:
                calib = data_dict['calib']
            if self.augment_randomly:
                data_dict = self.data_augmentor.forward_randomly(
                    data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
                )
            else:
                data_dict = self.data_augmentor.forward(
                    data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
                )

            if 'calib' in data_dict:
                data_dict['calib'] = calib
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(
                data_dict['gt_names'], self.class_names
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict['gt_names']],
                dtype=np.int32,
            )
            gt_boxes = np.concatenate(
                (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
                axis=1,
            )
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

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


# @staticmethod
# def collate_batch(batch_list, _unused=False):
#     data_dict = defaultdict(list)
#     for cur_sample in batch_list:
#         for key, val in cur_sample.items():
#             data_dict[key].append(val)
#     batch_size = len(batch_list)
#     ret = {}

#     for key, val in data_dict.items():
#         try:
#             if key in ['voxels', 'voxel_num_points']:
#                 ret[key] = np.concatenate(val, axis=0)
#             elif key in ['points', 'voxel_coords']:
#                 coors = []
#                 for i, coor in enumerate(val):
#                     coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
#                     coors.append(coor_pad)
#                 ret[key] = np.concatenate(coors, axis=0)
#             elif key in ['gt_boxes']:
#                 max_gt = max([len(x) for x in val])
#                 batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
#                 for k in range(batch_size):
#                     batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
#                 ret[key] = batch_gt_boxes3d
#             elif key in ['gt_scores']:
#                 max_gt = max([len(x) for x in val])
#                 batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
#                 for k in range(batch_size):
#                     batch_scores[k, :val[k].__len__()] = val[k]
#                 ret[key] = batch_scores
#             elif key in ['gt_boxes2d']:
#                 max_boxes = 0
#                 max_boxes = max([len(x) for x in val])
#                 batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
#                 for k in range(batch_size):
#                     if val[k].size > 0:
#                         batch_boxes2d[k, :val[k].__len__(), :] = val[k]
#                 ret[key] = batch_boxes2d
#             elif key in ["images", "depth_maps"]:
#                 # Get largest image size (H, W)
#                 max_h = 0
#                 max_w = 0
#                 for image in val:
#                     max_h = max(max_h, image.shape[0])
#                     max_w = max(max_w, image.shape[1])

#                 # Change size of images
#                 images = []
#                 for image in val:
#                     pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
#                     pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
#                     pad_width = (pad_h, pad_w)
#                     # Pad with nan, to be replaced later in the pipeline.
#                     pad_value = np.nan

#                     if key == "images":
#                         pad_width = (pad_h, pad_w, (0, 0))
#                     elif key == "depth_maps":
#                         pad_width = (pad_h, pad_w)

#                     image_pad = np.pad(image,
#                                         pad_width=pad_width,
#                                         mode='constant',
#                                         constant_values=pad_value)

#                     images.append(image_pad)
#                 ret[key] = np.stack(images, axis=0)
#             elif key in ['object_scale_noise', 'object_rotate_noise']:
#                     max_noise = max([len(x) for x in val])
#                     batch_noise = np.zeros((batch_size, max_noise), dtype=np.float32)
#                     for k in range(batch_size):
#                         batch_noise[k, :val[k].__len__()] = val[k]
#                     ret[key] = batch_noise
#             elif key in ['beam_labels']:
#                 continue
#             else:
#                 ret[key] = np.stack(val, axis=0)
#         except:
#             print('Error in collate_batch: key=%s' % key)
#             raise TypeError

#     ret['batch_size'] = batch_size
#     return ret
