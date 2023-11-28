import time

import numpy as np

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


def nms(original_boxes, iou_threshold):

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
