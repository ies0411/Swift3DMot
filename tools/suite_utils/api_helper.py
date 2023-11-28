import requests
import json
import math
import cv2
import numpy as np
import urllib
import os
from scipy.spatial.transform import Rotation
from pathlib import Path

API_HOST_URL = 'https://suite-api.dev.superb-ai.com'


def login(TENANT, EMAIL, PASSWORD, API_HOST=API_HOST_URL):
    r = requests.post(
        f'{API_HOST}/auth/system/tenants/login',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'tenant_id': TENANT, 'email': EMAIL, 'password': PASSWORD}),
    )
    r_data = r.json()['data']
    id_token, refresh_token = r_data['id_token'], r_data['refresh_token']

    return id_token, refresh_token


def get_project_by_name(id_token, project_name, API_HOST=API_HOST_URL):
    api_string = f'{API_HOST}/projects/name/{project_name}/'
    project = requests.get(
        api_string,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if project.status_code == 200:
        return project.json()
    else:
        return None


def get_project_by_id(id_token, project_id, API_HOST=API_HOST_URL):
    api_string = f'{API_HOST}/projects/{project_id}/'
    project = requests.get(
        api_string,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if project.status_code == 200:
        return project.json()
    else:
        return None


def get_label_list(
    id_token, project_id, options={}, page_size=100, API_HOST=API_HOST_URL
):
    api_string = f'{API_HOST}/projects/{project_id}/labels_id/?page_size={page_size}'
    if 'asset_key_in' in options:
        api_string += f'&asset_key_in[]={options["asset_key_in"]}'
    if 'last_updated_at_gte' in options:
        api_string += f'&last_updated_at_gte={options["last_update_at_gte"]}'
    if 'last_updated_at_lt' in options:
        api_string += f'&last_updated_at_lt={options["last_update_at_lt"]}'
    if 'tags' in options:
        all_tags_in_project_response = requests.get(
            f'{API_HOST}/projects/{project_id}/tags/',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {id_token}',
            },
        )
        if all_tags_in_project_response.status_code == 200:
            all_tags_in_project = all_tags_in_project_response.json()
        else:
            raise ValueError('error happens when get tags')
        for tag_name in options['tags']:
            for tag_info in all_tags_in_project:
                if tag_info['name'] == tag_name:
                    api_string += f'&tags_all[]={tag_info["id"]}'
                    break

    all_labels_info = []
    labels_info = requests.get(
        api_string,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    ).json()

    num_trial = math.ceil(labels_info['count'] / page_size)

    all_labels_info.extend(labels_info['results'])
    last_id = labels_info['results'][-1]['id']

    for i in range(num_trial - 1):
        labels_info = requests.get(
            api_string + f'&last_id={last_id}',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {id_token}',
            },
        ).json()
        all_labels_info.extend(labels_info['results'])
        last_id = labels_info['results'][-1]['id']

    return all_labels_info


def get_label(id_token, project_id, label_id, API_HOST=API_HOST_URL):
    read_label = requests.get(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if read_label.status_code != 200:
        return None

    return read_label.json()


def get_label_json(id_token, project_id, label_id, API_HOST=API_HOST_URL):
    result = {}
    read_label = requests.get(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if read_label.status_code != 200:
        return None
    result['label'] = read_label.json()

    read_info_url = requests.post(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/info/read-url/',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    ).json()
    info_json = requests.get(read_info_url['url'])
    if info_json.status_code == 200:
        result['label_info'] = info_json.json()

    return result


def get_label_by_name(
    id_token, project_id, asset_key_in, verbose=False, API_HOST=API_HOST_URL
):
    res = requests.get(
        f'{API_HOST}/projects/{project_id}/labels/?asset_key_icontains={asset_key_in}',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if verbose:
        print(res.status_code)
    if res.status_code != 200:
        return None

    return res.json()


def upload_label_info_json(
    id_token, project_id, label_id, label_info_to_upload, API_HOST=API_HOST_URL
):
    upload_url = requests.post(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/info/upload-url/',
        data=json.dumps({'file_size': len(json.dumps(label_info_to_upload))}),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if upload_url.status_code != 200:
        return None
    url = upload_url.json()["url"]
    upload_result = requests.put(url, data=json.dumps(label_info_to_upload))
    if upload_result.status_code == 200:
        return True
    return None


def upload_label(id_token, project_id, label_id, tags, API_HOST=API_HOST_URL):
    label_result = requests.patch(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/info/',
        data=json.dumps({"tags": tags}),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if label_result.status_code != 200:
        return None
    return True


def download_label(
    id_token,
    project_id,
    label_id,
    target_folder,
    label_only=False,
    API_HOST=API_HOST_URL,
):
    label = get_label(id_token, project_id, label_id)
    asset_id = label['asset']['id']
    asset_url_response = requests.post(
        f'{API_HOST}/assets/{asset_id}/read-signed-url/',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )

    if not (asset_url_response.ok):
        raise ValueError('cannot load asset urls')

    asset_url = asset_url_response.json()
    frame_urls = asset_url['url']['frame_urls']
    manifest_url = asset_url['url']['manifest_url']

    target_folder_path = Path(target_folder)
    if not target_folder_path.is_dir():
        target_folder_path.mkdir(parents=True, exist_ok=False)

    manifest_json = requests.get(manifest_url).json()
    with open(target_folder_path / 'manifest.json', 'w') as f:
        json.dump(manifest_json, f)
    result_info = []

    for frame_url in frame_urls:
        bin_url = frame_url['frame_url']
        pcd_filename = os.path.basename(bin_url.split('?')[0])
        urllib.request.urlretrieve(bin_url, str(target_folder_path / pcd_filename))

        image_urls = frame_url['image_urls']
        image_files = []
        for image_url in image_urls:
            filename = os.path.basename(image_url.split('?')[0])
            urllib.request.urlretrieve(image_url, str(target_folder_path / filename))
            image_files.append(filename)

        result_info.append({"pcd_file": pcd_filename, "image_files": image_files})

    with open(target_folder_path / 'result_info.json', 'w') as f:
        json.dump(result_info, f)

    if label_only:
        return

    read_url_response = requests.post(
        f'{API_HOST}/projects/{project_id}/labels/{label_id}/info/read-url/',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {id_token}',
        },
    )
    if read_url_response.ok:
        read_url = read_url_response.json()['label']
    else:
        raise ValueError('getting read url failed')

    info_json_response = requests.get(read_url)
    if info_json_response.ok:
        info_json = info_json_response.json()
        with open(target_folder_path / 'info.json', 'w') as f:
            json.dump(info_json, f)
