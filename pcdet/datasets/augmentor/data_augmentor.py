# from functools import partial

# import numpy as np

# from ...utils import common_utils
# from . import augmentor_utils, database_sampler


# class DataAugmentor(object):
#     def __init__(self, root_path, augmentor_configs, class_names, logger=None):
#         self.root_path = root_path
#         self.class_names = class_names
#         self.logger = logger

#         self.data_augmentor_queue = []
#         aug_config_list = (
#             augmentor_configs
#             if isinstance(augmentor_configs, list)
#             else augmentor_configs.AUG_CONFIG_LIST
#         )

#         for cur_cfg in aug_config_list:
#             if not isinstance(augmentor_configs, list):
#                 if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
#                     continue
#             cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
#             self.data_augmentor_queue.append(cur_augmentor)

#     def gt_sampling(self, config=None):
#         db_sampler = database_sampler.DataBaseSampler(
#             root_path=self.root_path,
#             sampler_cfg=config,
#             class_names=self.class_names,
#             logger=self.logger,
#         )
#         return db_sampler

#     def __getstate__(self):
#         d = dict(self.__dict__)
#         del d['logger']
#         return d

#     def __setstate__(self, d):
#         self.__dict__.update(d)

#     def random_world_flip(self, data_dict=None, config=None):
#         if data_dict is None:
#             return partial(self.random_world_flip, config=config)
#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
#         for cur_axis in config['ALONG_AXIS_LIST']:
#             assert cur_axis in ['x', 'y']
#             gt_boxes, points, enable = getattr(
#                 augmentor_utils, 'random_flip_along_%s' % cur_axis
#             )(gt_boxes, points, return_flip=True)
#             data_dict['flip_%s' % cur_axis] = enable
#             if 'roi_boxes' in data_dict.keys():
#                 num_frame, num_rois, dim = data_dict['roi_boxes'].shape
#                 roi_boxes, _, _ = getattr(
#                     augmentor_utils, 'random_flip_along_%s' % cur_axis
#                 )(
#                     data_dict['roi_boxes'].reshape(-1, dim),
#                     np.zeros([1, 3]),
#                     return_flip=True,
#                     enable=enable,
#                 )
#                 data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_world_rotation(self, data_dict=None, config=None):
#         if data_dict is None:
#             return partial(self.random_world_rotation, config=config)
#         rot_range = config['WORLD_ROT_ANGLE']
#         if not isinstance(rot_range, list):
#             rot_range = [-rot_range, rot_range]
#         gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
#             data_dict['gt_boxes'],
#             data_dict['points'],
#             rot_range=rot_range,
#             return_rot=True,
#         )
#         if 'roi_boxes' in data_dict.keys():
#             num_frame, num_rois, dim = data_dict['roi_boxes'].shape
#             roi_boxes, _, _ = augmentor_utils.global_rotation(
#                 data_dict['roi_boxes'].reshape(-1, dim),
#                 np.zeros([1, 3]),
#                 rot_range=rot_range,
#                 return_rot=True,
#                 noise_rotation=noise_rot,
#             )
#             data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         data_dict['noise_rot'] = noise_rot
#         return data_dict

#     def random_world_scaling(self, data_dict=None, config=None):
#         if data_dict is None:
#             return partial(self.random_world_scaling, config=config)

#         if 'roi_boxes' in data_dict.keys():
#             (
#                 gt_boxes,
#                 roi_boxes,
#                 points,
#                 noise_scale,
#             ) = augmentor_utils.global_scaling_with_roi_boxes(
#                 data_dict['gt_boxes'],
#                 data_dict['roi_boxes'],
#                 data_dict['points'],
#                 config['WORLD_SCALE_RANGE'],
#                 return_scale=True,
#             )
#             data_dict['roi_boxes'] = roi_boxes
#         else:
#             gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
#                 data_dict['gt_boxes'],
#                 data_dict['points'],
#                 config['WORLD_SCALE_RANGE'],
#                 return_scale=True,
#             )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         data_dict['noise_scale'] = noise_scale
#         return data_dict

#     def random_image_flip(self, data_dict=None, config=None):
#         if data_dict is None:
#             return partial(self.random_image_flip, config=config)
#         images = data_dict["images"]
#         depth_maps = data_dict["depth_maps"]
#         gt_boxes = data_dict['gt_boxes']
#         gt_boxes2d = data_dict["gt_boxes2d"]
#         calib = data_dict["calib"]
#         for cur_axis in config['ALONG_AXIS_LIST']:
#             assert cur_axis in ['horizontal']
#             images, depth_maps, gt_boxes = getattr(
#                 augmentor_utils, 'random_image_flip_%s' % cur_axis
#             )(
#                 images,
#                 depth_maps,
#                 gt_boxes,
#                 calib,
#             )

#         data_dict['images'] = images
#         data_dict['depth_maps'] = depth_maps
#         data_dict['gt_boxes'] = gt_boxes
#         return data_dict

#     def random_world_translation(self, data_dict=None, config=None):
#         if data_dict is None:
#             return partial(self.random_world_translation, config=config)
#         noise_translate_std = config['NOISE_TRANSLATE_STD']
#         assert len(noise_translate_std) == 3
#         noise_translate = np.array(
#             [
#                 np.random.normal(0, noise_translate_std[0], 1),
#                 np.random.normal(0, noise_translate_std[1], 1),
#                 np.random.normal(0, noise_translate_std[2], 1),
#             ],
#             dtype=np.float32,
#         ).T

#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
#         points[:, :3] += noise_translate
#         gt_boxes[:, :3] += noise_translate

#         if 'roi_boxes' in data_dict.keys():
#             data_dict['roi_boxes'][:, :3] += noise_translate

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_local_translation(self, data_dict=None, config=None):
#         """
#         Please check the correctness of it before using.
#         """
#         if data_dict is None:
#             return partial(self.random_local_translation, config=config)
#         offset_range = config['LOCAL_TRANSLATION_RANGE']
#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
#         for cur_axis in config['ALONG_AXIS_LIST']:
#             assert cur_axis in ['x', 'y', 'z']
#             gt_boxes, points = getattr(
#                 augmentor_utils, 'random_local_translation_along_%s' % cur_axis
#             )(
#                 gt_boxes,
#                 points,
#                 offset_range,
#             )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_local_rotation(self, data_dict=None, config=None):
#         """
#         Please check the correctness of it before using.
#         """
#         if data_dict is None:
#             return partial(self.random_local_rotation, config=config)
#         rot_range = config['LOCAL_ROT_ANGLE']
#         if not isinstance(rot_range, list):
#             rot_range = [-rot_range, rot_range]
#         gt_boxes, points = augmentor_utils.local_rotation(
#             data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
#         )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_local_scaling(self, data_dict=None, config=None):
#         """
#         Please check the correctness of it before using.
#         """
#         if data_dict is None:
#             return partial(self.random_local_scaling, config=config)
#         gt_boxes, points = augmentor_utils.local_scaling(
#             data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
#         )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_world_frustum_dropout(self, data_dict=None, config=None):
#         """
#         Please check the correctness of it before using.
#         """
#         if data_dict is None:
#             return partial(self.random_world_frustum_dropout, config=config)

#         intensity_range = config['INTENSITY_RANGE']
#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
#         for direction in config['DIRECTION']:
#             assert direction in ['top', 'bottom', 'left', 'right']
#             gt_boxes, points = getattr(
#                 augmentor_utils, 'global_frustum_dropout_%s' % direction
#             )(
#                 gt_boxes,
#                 points,
#                 intensity_range,
#             )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_local_frustum_dropout(self, data_dict=None, config=None):
#         """
#         Please check the correctness of it before using.
#         """
#         if data_dict is None:
#             return partial(self.random_local_frustum_dropout, config=config)

#         intensity_range = config['INTENSITY_RANGE']
#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
#         for direction in config['DIRECTION']:
#             assert direction in ['top', 'bottom', 'left', 'right']
#             gt_boxes, points = getattr(
#                 augmentor_utils, 'local_frustum_dropout_%s' % direction
#             )(
#                 gt_boxes,
#                 points,
#                 intensity_range,
#             )

#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def random_local_pyramid_aug(self, data_dict=None, config=None):
#         """
#         Refer to the paper:
#             SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
#         """
#         if data_dict is None:
#             return partial(self.random_local_pyramid_aug, config=config)

#         gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

#         gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(
#             gt_boxes, points, config['DROP_PROB']
#         )
#         gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(
#             gt_boxes,
#             points,
#             config['SPARSIFY_PROB'],
#             config['SPARSIFY_MAX_NUM'],
#             pyramids,
#         )
#         gt_boxes, points = augmentor_utils.local_pyramid_swap(
#             gt_boxes, points, config['SWAP_PROB'], config['SWAP_MAX_NUM'], pyramids
#         )
#         data_dict['gt_boxes'] = gt_boxes
#         data_dict['points'] = points
#         return data_dict

#     def forward(self, data_dict):
#         """
#         Args:
#             data_dict:
#                 points: (N, 3 + C_in)
#                 gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
#                 gt_names: optional, (N), string
#                 ...

#         Returns:
#         """
#         for cur_augmentor in self.data_augmentor_queue:
#             data_dict = cur_augmentor(data_dict=data_dict)

#         data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
#             data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
#         )
#         # if 'calib' in data_dict:
#         #     data_dict.pop('calib')
#         if 'road_plane' in data_dict:
#             data_dict.pop('road_plane')
#         if 'gt_boxes_mask' in data_dict:
#             gt_boxes_mask = data_dict['gt_boxes_mask']
#             data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
#             data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
#             if 'gt_boxes2d' in data_dict:
#                 data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

#             data_dict.pop('gt_boxes_mask')
#         return data_dict

import random
from functools import partial

import numpy as np
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, downsample_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(
        self,
        root_path,
        augmentor_configs,
        class_names,
        logger=None,
    ):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = {}
        aug_config_list = (
            augmentor_configs
            if isinstance(augmentor_configs, list)
            else augmentor_configs.AUG_CONFIG_LIST
        )

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)

            # self.data_augmentor_queue.append({cur_cfg.Name:cur_augmentor})
            self.data_augmentor_queue.update({cur_cfg.NAME: cur_augmentor})

    #####
    def get_polar_image(self, points):
        theta, phi = downsample_utils.compute_angles(points[:, :3])
        r = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
        polar_image = points.copy()
        polar_image[:, 0] = phi
        polar_image[:, 1] = theta
        polar_image[:, 2] = r
        return polar_image

    def label_point_cloud_beam(self, polar_image, points, beam=32):
        if polar_image.shape[0] <= beam:
            print("too small point cloud!")
            return np.arange(polar_image.shape[0])
        import time

        # start_time = time.time()
        beam_label, centroids = downsample_utils.beam_label(polar_image[:, 1], beam)
        # print(f'beam label : {time.time()-start_time}')
        idx = np.argsort(centroids)
        rev_idx = np.zeros_like(idx)
        for i, t in enumerate(idx):
            rev_idx[t] = i
        beam_label = rev_idx[beam_label]
        return beam_label

    def random_beam_upsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_upsample, config=config)
        points = data_dict['points']
        import time

        # start_time = time.time()
        polar_image = self.get_polar_image(points)
        # print(f'polar image : {time.time()-start_time}')

        beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])

        new_pcs = [points]
        phi = polar_image[:, 0]
        # start_time = time.time()
        for i in range(config['BEAM'] - 1):
            if np.random.rand() < config['BEAM_PROB'][i]:
                cur_beam_mask = beam_label == i
                next_beam_mask = beam_label == i + 1
                delta_phi = np.abs(
                    phi[cur_beam_mask, np.newaxis] - phi[np.newaxis, next_beam_mask]
                )
                corr_idx = np.argmin(delta_phi, 1)
                min_delta = np.min(delta_phi, 1)
                mask = min_delta < config['PHI_THRESHOLD']
                cur_beam = polar_image[cur_beam_mask][mask]
                next_beam = polar_image[next_beam_mask][corr_idx[mask]]
                new_beam = (cur_beam + next_beam) / 2
                new_pc = new_beam.copy()
                new_pc[:, 0] = (
                    np.cos(new_beam[:, 1]) * np.cos(new_beam[:, 0]) * new_beam[:, 2]
                )
                new_pc[:, 1] = (
                    np.cos(new_beam[:, 1]) * np.sin(new_beam[:, 0]) * new_beam[:, 2]
                )
                new_pc[:, 2] = np.sin(new_beam[:, 1]) * new_beam[:, 2]
                new_pcs.append(new_pc)
        # print(f'up : {time.time()-start_time}')
        data_dict['points'] = np.concatenate(new_pcs, 0)
        # print(f'label : {time.time()-start_time}')
        return data_dict

    def random_beam_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            # return data_dict
            return partial(self.random_beam_downsample, config=config)
        # print("downsample")
        # import time

        # start_time = time.time()
        points = data_dict['points']
        if 'beam_labels' in data_dict:  # for waymo and kitti datasets
            beam_label = data_dict['beam_labels']
        else:
            polar_image = self.get_polar_image(points)
            beam_label = self.label_point_cloud_beam(
                polar_image, points, config['BEAM']
            )
        beam_mask = np.random.rand(config['BEAM']) < config['BEAM_PROB']
        points_mask = beam_mask[beam_label]
        data_dict['points'] = points[points_mask]
        if config.get('FILTER_GT_BOXES', None):
            num_points_in_gt = (
                roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7]),
                )
                .numpy()
                .sum(axis=1)
            )

            mask = num_points_in_gt >= config.get('MIN_POINTS_OF_GT', 1)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]
            if 'gt_boxes_mask' in data_dict:
                data_dict['gt_boxes_mask'] = data_dict['gt_boxes_mask'][mask]
        # print(f'downsample : {time.time()-start_time}')
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes, object_scale_noise = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'],
            data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE'],
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['object_scale_noise'] = object_scale_noise
        return data_dict

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points, object_rotate_noise = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50,
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['object_rotate_noise'] = object_rotate_noise
        return data_dict

    ####
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger,
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        # print("world flip")
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(
                augmentor_utils, 'random_flip_along_%s' % cur_axis
            )(gt_boxes, points, return_flip=True)
            data_dict['flip_%s' % cur_axis] = enable
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois, dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(
                    augmentor_utils, 'random_flip_along_%s' % cur_axis
                )(
                    data_dict['roi_boxes'].reshape(-1, dim),
                    np.zeros([1, 3]),
                    return_flip=True,
                    enable=enable,
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = augmentor_utils.global_rotation(
            data_dict['gt_boxes'],
            data_dict['points'],
            rot_range=rot_range,
            return_rot=True,
        )
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois, dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
                data_dict['roi_boxes'].reshape(-1, dim),
                np.zeros([1, 3]),
                rot_range=rot_range,
                return_rot=True,
                noise_rotation=noise_rot,
            )
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois, dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_rot'] = noise_rot
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        if 'roi_boxes' in data_dict.keys():
            (
                gt_boxes,
                roi_boxes,
                points,
                noise_scale,
            ) = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'],
                data_dict['roi_boxes'],
                data_dict['points'],
                config['WORLD_SCALE_RANGE'],
                return_scale=True,
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'],
                data_dict['points'],
                config['WORLD_SCALE_RANGE'],
                return_scale=True,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(
                augmentor_utils, 'random_image_flip_%s' % cur_axis
            )(
                images,
                depth_maps,
                gt_boxes,
                calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array(
            [
                np.random.normal(0, noise_translate_std[0], 1),
                np.random.normal(0, noise_translate_std[1], 1),
                np.random.normal(0, noise_translate_std[2], 1),
            ],
            dtype=np.float32,
        ).T

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate

        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :3] += noise_translate

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(
                augmentor_utils, 'random_local_translation_along_%s' % cur_axis
            )(
                gt_boxes,
                points,
                offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(
                augmentor_utils, 'global_frustum_dropout_%s' % direction
            )(
                gt_boxes,
                points,
                intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(
                augmentor_utils, 'local_frustum_dropout_%s' % direction
            )(
                gt_boxes,
                points,
                intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(
            gt_boxes, points, config['DROP_PROB']
        )
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(
            gt_boxes,
            points,
            config['SPARSIFY_PROB'],
            config['SPARSIFY_MAX_NUM'],
            pyramids,
        )
        gt_boxes, points = augmentor_utils.local_pyramid_swap(
            gt_boxes, points, config['SWAP_PROB'], config['SWAP_MAX_NUM'], pyramids
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward_randomly(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # print(f'aug len : {len(self.data_augmentor_queue)}')

        # for cur_augmentor in self.data_augmentor_queue:
        for name, cur_augmentor in self.data_augmentor_queue.items():
            if name == "random_beam_downsample" or name == "random_beam_upsample":
                if random.choices(range(1, 10))[0] != 1:
                    continue
            # else:
            #     if random.choices(range(1, 2))[0] != 1:
            #         continue
            data_dict = cur_augmentor(data_dict=data_dict)

        #             value = car.get("company")

        # if value == None:
        # 	print("Key not exist!")
        # else:
        # 	print("Key exist! The value is " + car["name"])
        # if idx == 0 or idx == 1:
        #     if random.choices(range(1, 10))[0] != 1:
        #         continue
        # else
        # data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # print(f'aug len : {len(self.data_augmentor_queue)}')
        # for cur_augmentor in self.data_augmentor_queue:
        #     data_dict = cur_augmentor(data_dict=data_dict)

        for _, cur_augmentor in self.data_augmentor_queue.items():
            # if name == "random_beam_downsample" or name == "random_beam_upsample":
            #     if random.choices(range(1, 10))[0] != 1:
            #         continue
            # else:
            #     if random.choices(range(1, 2))[0] != 1:
            #         continue
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # if 'calib' in data_dict:
        #     data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
