import yaml
import json
import requests

import os
import glob
from natsort import natsorted
import numpy as np
import math

# import quaternion
import uuid
import copy

from scipy.spatial.transform import Rotation

# import cv2
from utils.calibration_utils.kitti import load_intrinsic as kitty_intrinsic
from utils.calibration_utils.kitti import (
    load_cam_velo_extrinsic as kitty_cam_lidar_extrinsic,
)

# from pcdet.utils import common_utils

# logger = common_utils.create_logger()


class Suite(object):
    """docstring for Suite."""

    def __init__(self, config):
        super(Suite, self).__init__()
        self.suite_url = config["suite_url"]
        self.tenant = config["tenant"]
        self.email = config["email"]
        self.password = config["password"]
        self.target_folder = config["target_folder"]

        self.image_width = config["image_width"]
        self.image_height = config["image_height"]
        self.calib_data_type = config["calib_data_type"]
        self.data_name = config["data_name"]
        self.folder_name = config["folder_name"]
        self.bin_lidar_files = natsorted(
            glob.glob(os.path.join(self.target_folder, "pcd", "*.bin"))
        )
        self.image_files = natsorted(
            glob.glob(os.path.join(self.target_folder, "image", "*.png"))
        )
        self.project_name = config["project_name"]
        self.camera_type = config["camera_type"]
        self.manifest_template = {
            "key": self.data_name,
            "manifest": {
                "prefix": "./",
                "frame_count": len(self.bin_lidar_files),  # lidar 갯수 기준
                "frames": [],
            },
        }
        self.object_template = {
            "id": str(uuid.uuid4()),  # 이 객체의 id
            "class_id": "0",  # 이 객체가 속하는 project의 label interface의 class id
            "tracking_id": 0,
            "class_name": "None",
            "annotation_type": "cuboid",  # 3d 에서는 이것밖에없음 현재는
            "frames": [],
            "properties": [],
        }
        self.object_frame_template = {
            "num": 0,  # frame index 0-base
            "properties": [],
            "annotation": {
                "coord": {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "rotation_quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
                    "size": {"x": 1, "y": 1, "z": 1},
                },
                "meta": {"visible": True, "alpha": 1, "color": "#FF625A"},
            },
        }

    def set_manifest_frame_info(
        self, idx, intrinsic_matrix, translation, extrinsic_quaternion
    ):
        images = [
            {
                "image_path": str(self.image_files[idx]),
                "timestamp": 0,
                "camera_model": self.camera_type,
                "cx": intrinsic_matrix[0][2],
                "cy": intrinsic_matrix[1][2],
                "fx": intrinsic_matrix[0][0],
                "fy": intrinsic_matrix[1][1],
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "k4": 0,
                "p1": 0,
                "p2": 0,
                "skew": 0,
                "new_camera_matrix": {
                    "cx": intrinsic_matrix[0][2],
                    "cy": intrinsic_matrix[1][2],
                    "fx": intrinsic_matrix[0][0],
                    "fy": intrinsic_matrix[1][1],
                    "skew": 0,
                },
                "position": {
                    "x": translation[0],
                    "y": translation[1],
                    "z": translation[2],
                },
                "heading": {
                    "qw": extrinsic_quaternion[3],
                    "qx": extrinsic_quaternion[0],
                    "qy": extrinsic_quaternion[1],
                    "qz": extrinsic_quaternion[2],
                },
            }
        ]

        manifest_frame_info = {
            "frame_number": idx + 1,
            "timestamp": 0,
            "frame_path": str(self.bin_lidar_files[idx]),
            "meta": {
                "ego_vehicle_pose": {
                    "heading": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            },
            "images": images,
        }
        return manifest_frame_info

    def upload_data(self):
        try:
            r = requests.post(
                f"{self.suite_url}/auth/login",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "tenant_id": self.tenant,
                        "email": self.email,
                        "password": self.password,
                    }
                ),
            )
        except Exception as e:
            print(e)
            return False
        r_data = r.json()["data"]

        id_token, refresh_token = r_data["id_token"], r_data["refresh_token"]
        # print(id_token)
        calib_file = os.path.join(self.target_folder, "calib.txt")
        if self.calib_data_type == "KITTI":
            intrinsic_matrix = kitty_intrinsic(calib_file)
            # print(intrinsic_matrix)
            extrinsic_matrix = kitty_cam_lidar_extrinsic(calib_file)
            # print(extrinsic_matrix)
            rotation = extrinsic_matrix[:, :3]
            translation = np.matmul(rotation.T, -extrinsic_matrix[:, 3])
            heading = Rotation.from_matrix(rotation.T)
            extrinsic_quaternion = heading.as_quat()
        elif self.calib_data_type == "NUSCENES":
            print("todo")
            # TODO : nuscenes calib
        else:
            print("please fille calib data type")
            return False

        for idx in range(len(self.bin_lidar_files)):
            manifest_frame_info = self.set_manifest_frame_info(
                idx, intrinsic_matrix, translation, extrinsic_quaternion
            )

            self.manifest_template["manifest"]["frames"].append(manifest_frame_info)

        manifest_file_size = len(json.dumps(self.manifest_template))
        # print(f"size : {manifest_file_size}")
        frame_infos = []
        frame_count = 0
        total_count = 0

        files_path = {"manifest_path": "manifest.json", "frame_paths": []}

        for frame_info in self.manifest_template["manifest"]["frames"]:
            frame_path = frame_info["frame_path"]
            frame_file_size = os.path.getsize(frame_path)
            frame_number = frame_info["frame_number"]
            image_count = 0
            image_infos = []

            frame_path_info = {"frame_path": frame_path, "image_paths": []}
            for image_info in frame_info["images"]:
                image_path = image_info["image_path"]
                image_file_size = os.path.getsize(image_path)
                frame_path_info["image_paths"].append(image_path)
                image_infos.append(
                    {"image_file_name": image_path, "image_file_size": image_file_size}
                )

                image_count += 1
                total_count += 1

            frame_infos.append(
                {
                    "frame_number": frame_number,
                    "frame_file_name": frame_path,
                    "frame_file_size": frame_file_size,
                    "image_count": image_count,
                    "image_infos": image_infos,
                }
            )
            files_path["frame_paths"].append(frame_path_info)
            frame_count += 1
            total_count += 1

        pointclouds_data = {
            "key": self.data_name,  # 직접 올릴 때 사용하는 이름
            "group": self.folder_name,  # 직접 올릴 때 사용하는 폴더
            "manifest_file_name": "manifest.json",
            "manifest_file_size": manifest_file_size,
            "sequence_number": 1,  # 여러 개 시퀀스를 올릴떄는 의미가 있지만, for loop돌면서 sequence별로 따로 올리는게 좋음
            "frame_count": frame_count,
            "total_file_count": total_count,  # image 수 + lidar 수 + manifest
            "frame_infos": frame_infos,
        }
        # API를 쏠 때 덧붙이는 정보 id token을 통해서 인증정보 붙여줌
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + id_token,
        }

        # Asset을 생성
        try:
            pointclouds_presigned_url_response = requests.post(
                self.suite_url + "/assets/typed/pointclouds-presigned-url/",
                data=json.dumps(pointclouds_data),
                headers=headers,
            )
        except Exception as e:
            print(e)
            return False
        asset_id = pointclouds_presigned_url_response.json()["id"]

        upload_url_response = requests.post(
            self.suite_url + f"/assets/{asset_id}/upload-url/", headers=headers
        )
        upload_url = upload_url_response.json()["url"]
        upload_url.keys()
        upload_url["frame_urls"]  # 10개짜리 길이의 list
        upload_url["frame_urls"][0].keys()

        upload_manifest_url = upload_url["manifest_url"]
        manifest_data = json.dumps(self.manifest_template).encode()
        s3_upload_response = requests.put(upload_manifest_url, data=manifest_data)

        for frame_path, frame_url in zip(
            files_path["frame_paths"], upload_url["frame_urls"]
        ):
            with open(frame_path["frame_path"], "rb") as data:
                s3_upload_response = requests.put(frame_url["frame_url"], data=data)

            for image_path, image_url in zip(
                frame_path["image_paths"], frame_url["image_urls"]
            ):
                with open(image_path, "rb") as data:
                    s3_upload_response = requests.put(image_url, data=data)

    def upload_label(self, pred_label_dict):
        r = requests.post(
            f"{self.suite_url}/auth/login",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "tenant_id": self.tenant,
                    "email": self.email,
                    "password": self.password,
                }
            ),
        )
        r_data = r.json()["data"]
        id_token, refresh_token = r_data["id_token"], r_data["refresh_token"]
        print(id_token)
        # label_info_to_upload = {
        #     "version": "0.6.3",
        #     "meta": {
        #         "image_info": {},
        #         "edit_info": {
        #             "brightness": 0,
        #             "contrast": 0,
        #             "elapsed_time": 0,
        #             "objects": [],
        #             "canvas_scale": 1,
        #             "timeline_scale": 1,
        #         },
        #     },
        #     "result": {"objects": [], "categories": {"frames": [], "properties": []}},
        #     "tags": {
        #         "classes_id": [],
        #         "categories_id": [],
        #         "class": [],
        #         "classes_count": [],
        #         "time_spent": 0,
        #     },
        # }
        # # 원하는 프로젝트 정보 가져오기
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": "Bearer " + id_token,
        # }

        # project_info = requests.post(
        #     self.suite_url + "/projects/name/" + self.project_name, headers=headers
        # )

        # project_id = project_info.json()["id"]
        # # print(pred_label_dict)
        # # for idx, result in enumerate(result_list[0]):
        # for class_key, values in pred_label_dict.items():
        #     for idx, pred_list in enumerate(values):
        #         if pred_list.size == 0:
        #             continue
        #         for pred in pred_list:
        #             if len(pred) == 0:
        #                 continue

        #             object_dict = copy.deepcopy(self.object_template)
        #             object_dict["class_id"] = project_id
        #             object_dict["tracking_id"] = pred[7]
        #             object_dict["class_name"] = class_key
        #             object_dict["id"] = str(uuid.uuid4())
        #             object_frame_dict = copy.deepcopy(self.object_frame_template)
        #             object_frame_dict["annotation"]["coord"]["position"]["x"] = pred[3]
        #             object_frame_dict["annotation"]["coord"]["position"]["y"] = pred[4]
        #             object_frame_dict["annotation"]["coord"]["position"]["z"] = pred[5]
        #             object_frame_dict["annotation"]["coord"]["size"]["x"] = pred[0]
        #             object_frame_dict["annotation"]["coord"]["size"]["y"] = pred[1]
        #             object_frame_dict["annotation"]["coord"]["size"]["z"] = pred[2]

        #             theta = pred[6]
        #             euler_obj = [0, 0, theta]
        #             quat_obj = quaternion.from_euler_angles(euler_obj)
        #             object_frame_dict["annotation"]["coord"]["rotation_quaternion"][
        #                 "x"
        #             ] = quat_obj.x
        #             object_frame_dict["annotation"]["coord"]["rotation_quaternion"][
        #                 "y"
        #             ] = quat_obj.y
        #             object_frame_dict["annotation"]["coord"]["rotation_quaternion"][
        #                 "z"
        #             ] = quat_obj.z
        #             object_frame_dict["annotation"]["coord"]["rotation_quaternion"][
        #                 "w"
        #             ] = quat_obj.w
        #             object_dict["frames"].append(object_frame_dict)
        #             label_info_to_upload["result"]["objects"].append(object_dict)

        # class_to_class_id = {
        #     class_name: project_id,
        # }
        # class_to_class_color = {
        #     class_name: "#FF625A",
        # }

        # classes_id = set()
        # classes = set()
        # classes_count = {}
        # for obj in label_info_to_upload["result"]["objects"]:
        #     classes_id.add(obj["class_id"])
        #     classes.add(obj["class_name"])
        # if obj["class_name"] in classes_count:
        #     classes_count[obj["class_name"]] += 1
        # else:
        #     classes_count[obj["class_name"]] = 1
        # label_info_to_upload["tags"]["classes_id"] = list(classes_id)
        # label_info_to_upload["tags"]["class"] = list(classes)
        # label_info_to_upload["tags"]["classes_count"] = [
        #     {"id": class_to_class_id[k], "name": k, "count": v}
        #     for k, v in classes_count.items()
        # ]

        # asset_key_in = "my_lidar"

        # res = requests.get(
        #     f"{suite_url}/projects/{project_id}/labels/?asset_key_icontains={asset_key_in}",
        #     headers={
        #         "Content-Type": "application/json",
        #         "Authorization": f"Bearer {id_token}",
        #     },
        # )

        # label_id = res.json()["results"][0]["id"]

        # label_result = requests.patch(
        #     f"{suite_url}/projects/{project_id}/labels/{label_id}/info/",
        #     data=json.dumps({"tags": label_info_to_upload["tags"]}),
        #     headers={
        #         "Content-Type": "application/json",
        #         "Authorization": f"Bearer {id_token}",
        #     },
        # )

        # upload_url = requests.post(
        #     f"{suite_url}/projects/{project_id}/labels/{label_id}/info/upload-url/",
        #     data=json.dumps({"file_size": len(json.dumps(label_info_to_upload))}),
        #     headers={
        #         "Content-Type": "application/json",
        #         "Authorization": f"Bearer {id_token}",
        #     },
        # )
        # url = upload_url.json()["url"]

        # upload_result = requests.put(url, data=json.dumps(label_info_to_upload))
        # upload_result.status_code
