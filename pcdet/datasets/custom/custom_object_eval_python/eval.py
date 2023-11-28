import numpy as np

# https://github.com/open-mmlab/OpenPCDet/issues/1236
# https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/Metric_3d/#average-precision-kitti


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    metric: eval type. 0: bbox, 1: bev, 2: 3d
    min_overlaps: float, min overlap. format: [num_overlap, metric, class].
    """
    # name classes           car  ped  cyc  van  per  tru  mast
    # int classes:            0    1    2    3    4    5    6
    # bbox     x    x    x
    # bev      x    x    x
    # 3d       x    x    x
    # overlap_0_7 = np.array([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7]])
    # overlap_0_5 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    class_n = len(current_classes)
    overlap_0_7 = 0.7*np.ones(shape=(class_n,class_n),dtype=np.float32)
    overlap_0_5 = 0.5*np.ones(shape=(class_n,class_n),dtype=np.float32)
    min_overlaps = np.stack([overlap_0_5, overlap_0_7], axis=0)  # [2, 3, 5]
    # change here
    # class_to_name = {
    #     0: 'type1',  # add to overlap_0_7 (also to clean_data())
    #     1: 'type2',
    #     2: 'type3',
    # }
    class_to_name={}
    for i in range(class_n):
      class_to_name[i] = current_classes[i]
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))

            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                if i == 0:
                   ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                   ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                   ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            if i == 0:
                # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
                # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
                # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
                # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
                # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
                # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
                # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
                # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
                # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict

def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['type1', 'type2', 'type3']  # change here
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]


# def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
#     overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
#                              0.5, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
#                             [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
#     overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
#                              0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
#                             [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
#     min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
#     class_to_name = {
#         0: 'Car',
#         1: 'Pedestrian',
#         2: 'Cyclist',
#         3: 'Van',
#         4: 'Person_sitting',
#         5: 'Truck'
#     }
#     name_to_class = {v: n for n, v in class_to_name.items()}
#     if not isinstance(current_classes, (list, tuple)):
#         current_classes = [current_classes]
#     current_classes_int = []
#     for curcls in current_classes:
#         if isinstance(curcls, str):
#             current_classes_int.append(name_to_class[curcls])
#         else:
#             current_classes_int.append(curcls)
#     current_classes = current_classes_int
#     min_overlaps = min_overlaps[:, :, current_classes]
#     result = ''
#     # check whether alpha is valid
#     compute_aos = False
#     for anno in dt_annos:
#         if anno['alpha'].shape[0] != 0:
#             if anno['alpha'][0] != -10:
#                 compute_aos = True
#             break
#     mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
#         gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

#     ret_dict = {}
#     for j, curcls in enumerate(current_classes):
#         # mAP threshold array: [num_minoverlap, metric, class]
#         # mAP result: [num_class, num_diff, num_minoverlap]
#         for i in range(min_overlaps.shape[0]):
#             result += print_str(
#                 (f"{class_to_name[curcls]} "
#                  "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
#             result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
#                                  f"{mAPbbox[j, 1, i]:.4f}, "
#                                  f"{mAPbbox[j, 2, i]:.4f}"))
#             result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
#                                  f"{mAPbev[j, 1, i]:.4f}, "
#                                  f"{mAPbev[j, 2, i]:.4f}"))
#             result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
#                                  f"{mAP3d[j, 1, i]:.4f}, "
#                                  f"{mAP3d[j, 2, i]:.4f}"))

#             if compute_aos:
#                 result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
#                                      f"{mAPaos[j, 1, i]:.2f}, "
#                                      f"{mAPaos[j, 2, i]:.2f}"))
#                 # if i == 0:
#                    # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
#                    # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
#                    # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]

#             result += print_str(
#                 (f"{class_to_name[curcls]} "
#                  "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
#             result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
#                                  f"{mAPbbox_R40[j, 1, i]:.4f}, "
#                                  f"{mAPbbox_R40[j, 2, i]:.4f}"))
#             result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
#                                  f"{mAPbev_R40[j, 1, i]:.4f}, "
#                                  f"{mAPbev_R40[j, 2, i]:.4f}"))
#             result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
#                                  f"{mAP3d_R40[j, 1, i]:.4f}, "
#                                  f"{mAP3d_R40[j, 2, i]:.4f}"))
#             if compute_aos:
#                 result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
#                                      f"{mAPaos_R40[j, 1, i]:.2f}, "
#                                      f"{mAPaos_R40[j, 2, i]:.2f}"))
#                 if i == 0:
#                    ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
#                    ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
#                    ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

#             if i == 0:
#                 # ret_dict['%s_3d/easy' % class_to_name[curcls]] = mAP3d[j, 0, 0]
#                 # ret_dict['%s_3d/moderate' % class_to_name[curcls]] = mAP3d[j, 1, 0]
#                 # ret_dict['%s_3d/hard' % class_to_name[curcls]] = mAP3d[j, 2, 0]
#                 # ret_dict['%s_bev/easy' % class_to_name[curcls]] = mAPbev[j, 0, 0]
#                 # ret_dict['%s_bev/moderate' % class_to_name[curcls]] = mAPbev[j, 1, 0]
#                 # ret_dict['%s_bev/hard' % class_to_name[curcls]] = mAPbev[j, 2, 0]
#                 # ret_dict['%s_image/easy' % class_to_name[curcls]] = mAPbbox[j, 0, 0]
#                 # ret_dict['%s_image/moderate' % class_to_name[curcls]] = mAPbbox[j, 1, 0]
#                 # ret_dict['%s_image/hard' % class_to_name[curcls]] = mAPbbox[j, 2, 0]

#                 ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
#                 ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
#                 ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
#                 ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
#                 ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
#                 ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
#                 ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
#                 ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
#                 ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

#     return result, ret_dict
