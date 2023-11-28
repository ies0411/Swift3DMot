import numpy as np

# from pcdet.utils import common_utils

# logger = common_utils.create_logger()


def load_intrinsic(path):
    intrinsic_matrix = None
    with open(path) as f:
        file = f.readlines()
        for line in file:
            try:
                (key, val) = line.split(":", 1)
                if key == "P2":
                    intrinsic_matrix = np.fromstring(val, sep=" ")
                    intrinsic_matrix = intrinsic_matrix.reshape(3, 4)
            except Exception as e:
                print(e)
    return intrinsic_matrix


def load_cam_velo_extrinsic(path):
    RT = None
    with open(path) as f:
        file = f.readlines()
        for line in file:
            try:
                (key, val) = line.split(":", 1)
                if key == "Tr_velo_cam":
                    RT = np.fromstring(val, sep=" ")
                    RT = RT.reshape(3, 4)
            except Exception as e:
                print(e)
    return RT