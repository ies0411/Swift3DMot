import argparse
import os

import natsort
import numpy as np
# from pcdet.config import cfg
from pcdet.utils import common_utils
from tracking_modules.io import save_results
from tracking_modules.model import SWIFT3DMOT
from tracking_modules.utils import Config

# import time



def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--tracking_config",
        type=str,
        default="./tracking_modules/configs/config.yml",
        help="tracking config file path",
    )
    args = parser.parse_args()
    tracking_cfg = Config(args.tracking_config)
    return args, tracking_cfg

# save_trk(frame_idx, tracking_result,tracking_cfg)

def save_trk(group_idx, frame_idx, tracking_results,tracking_cfg):
    if not os.path.exists(tracking_cfg.tracking_results_path):
        os.makedirs(tracking_cfg.tracking_results_path)
    if not os.path.exists(os.path.join(tracking_cfg.tracking_results_path,str(group_idx).zfill(4)+".txt")):
        f = open(os.path.join(tracking_cfg.tracking_results_path,str(group_idx).zfill(4)+".txt"), 'w')
    else:
        f = open(os.path.join(tracking_cfg.tracking_results_path,str(group_idx).zfill(4)+".txt"), 'a')
    # 6 1 Car -1 -1 -10 374.1452 164.7073 448.1982 204.0086 2.0245 1.8002 3.9769 -10.8040 1.6052 39.3477 -20.0119 1.2941
    for tracking_result in tracking_results:
        print(tracking_result)
        f.write(f"{frame_idx} {tracking_result[-1]} {tracking_cfg.tracking_type} {-1} {-1} {-10} {374.1452} {164.7073} {448.1982} {204.0086} {tracking_result[0]} {tracking_result[1]} {tracking_result[2]} {tracking_result[3]} {tracking_result[4]} {tracking_result[5]} {tracking_result[6]} {tracking_result[7]}\n")
    f.close()

def main():
    args, tracking_cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------SWIFT 3DMOT-------------------------")
    tracking_results = []

    ID_start = 1
    id_max = 0

    det_file_list = os.listdir(tracking_cfg.save_path)
    det_file_list = natsort.natsorted(det_file_list)
    for det_file in det_file_list:
        tracker = SWIFT3DMOT(ID_init=ID_start)
        # order, class, img_x, img_y, dx,dy,dz,x,y,z,rotation,score,thres
        frame_idx = 0
        with open(os.path.join(tracking_cfg.save_path,det_file)) as f:
            det_frames = []
            for line in f.readlines():
                if frame_idx == int(line[0]):
                    one_det = line.split(' ')
                    det = [float(one_det[7]), float(one_det[8]),  float(one_det[9]),  float(one_det[4]),  float(one_det[5]),  float(one_det[6]),  float(one_det[10]),  float(one_det[11]),  float(one_det[12])]
                    det_frames.append(det)

                else:
                    # dx,dy,dz,x,y,z,yaw,id
                    # 6 1 Car -1 -1 -10 374.1452 164.7073 448.1982 204.0086 2.0245 1.8002 3.9769 -10.8040 1.6052 39.3477 -20.0119 1.2941
                    # frame, id, Car, -1, -1, -10(alpha) 374.1452 164.7073 448.1982 204.0086, dx,dy,dz,x,y,z,rotation, score
                    tracking_result, affi = tracker.track(det_frames)
                    # print(tracking_result[0].tolist())
                    save_trk(int(det_file.split('.')[0]), frame_idx, tracking_result[0].tolist(),tracking_cfg)
                    det_frames = []
                    frame_idx += 1
                    one_det = line.split(' ')
                    det = [float(one_det[7]), float(one_det[8]),  float(one_det[9]),  float(one_det[4]),  float(one_det[5]),  float(one_det[6]),  float(one_det[10]),  float(one_det[11]),  float(one_det[12])]
                    det_frames.append(det)




            # print(tracking_result)
# 6 1 Car -1 -1 -10 374.1452 164.7073 448.1982 204.0086 2.0245 1.8002 3.9769 -10.8040 1.6052 39.3477 -20.0119 1.2941

    # for data_dir in data_dir_list:
    #     if int(data_dir) in tracking_seqs:
    #         file_name_list = os.listdir(os.path.join(root_folder,data_dir))
    #         file_name_list = natsort.natsorted(file_name_list)
    #         for file_name in file_name_list:
    #             all_data_list.append(os.path.join(root_folder,data_dir,file_name))
    # return all_data_list
    # det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    # for label, pred_bboxes in tracking_info_data["bbox"].items():
    #     converted_bbox = []
    #     tracker = SWIFT3DMOT(ID_init=ID_start)
    #     for idx, pred_bbox in enumerate(pred_bboxes):
    #         for order, bbox in enumerate(pred_bbox):
    #             bbox.append(tracking_info_data["score"][label][idx][order])
    #             bbox.append(0.5)
    #             converted_bbox.append(bbox)
    #         tracking_result, affi = tracker.track(converted_bbox)
    #         print(f'tracking_result : {tracking_result}')
    #         convertformat_det_to_track(tracking_result)
    #         save_trk_file = os.path.join(args.tracking_output_dir, '%06d.txt' % idx)
    #         save_trk_file = open(save_trk_file, 'w')
    #         for result_tmp in tracking_result:				# N x 15
    #             save_results(result_tmp, save_trk_file, args.mot_output_dir, 0, idx, 0.5)
    #         save_trk_file.close()
    #         tracking_result = tracking_result[0]
    #         if len(tracking_result) != 0:
    #             id_max = max(id_max, tracking_result[0][7])
    #     ID_start = id_max + 1
    # tracking_results.append(tracking_result.tolist())

    # for frame, tracking_result in enumerate(tracking_results):
    #     save_trk_file = os.path.join(args.tracking_output_dir, '%06d.txt' % frame)
    #     save_trk_file = open(save_trk_file, 'w')
    #     for result_tmp in tracking_result:				# N x 15
    #         save_results(result_tmp, save_trk_file, eval_file_dict, det_id2str, frame, cfg.score_threshold)
    #     save_trk_file.close()



if __name__ == "__main__":
    main()
