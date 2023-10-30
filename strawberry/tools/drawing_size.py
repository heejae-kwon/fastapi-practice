# %%
import os
import zipfile
import json
import copy
import cv2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from PIL import Image
from plyfile import PlyData


# 데이터 디렉토리 경로 지정


def keypoint(id, output, image_dir):
    with open(output, 'r') as f:
        keypoint_json = json.load(f)

    image_id = keypoint_json[id]['image_id']
    keypoints = keypoint_json[id]['keypoints']

    if image_id < 10:
        img = cv2.imread('{0}/00000{1}.jpg'.format(image_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif image_id < 100:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif image_id < 1000:
        img = cv2.imread('{0}/000{1}.jpg'.format(image_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif image_id < 10000:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    kpt = keypoints[:12]
    line_k = []

    for k in range(0, len(kpt), 3):
        x = int(kpt[k])
        y = int(kpt[k+1])
        line_k.extend([x, y])
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

    return (img, image_id, line_k)

# 선 그리는 함수


def DrawLine(img, line_k):
    cv2.line(img, (line_k[0], line_k[1]), (line_k[4], line_k[5]),
             (0, 255, 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (line_k[2], line_k[3]), (line_k[6], line_k[7]),
             (0, 255, 0), thickness=6, lineType=cv2.LINE_AA)
    return (img)

# text 함수


def Text(img, size, id):
    cv2.putText(img, '{0}cm'.format(round(size['length'][id]['height']/10, 2)), (round(
        (line_k[0]+line_k[4])/2), round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    cv2.putText(img, '{0}cm'.format(round(size['length'][id]['width']/10, 2)),
                (line_k[2], line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
    plt.imshow(img)
    return (img)


# %%
output_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/output/straw_try/pose_hrnet/keypoint_strawberry'
output_est = os.path.join(output_dir, os.listdir(
    output_dir)[-1], 'results/keypoints_validation_results_0.json')

valid_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/strawberry/validation'
valid_json_file = os.path.join(valid_dir, 'val-coco_style.json')
output = os.path.join(valid_dir, 'keypoints_validation_results_0.json')

img_path = os.path.join(valid_dir, 'image')
ply_path = os.path.join(valid_dir, 'ply')

# %%
img, img_id, line_k = keypoint(15, output_est, img_path)
img = DrawLine(img, line_k)
# img = Text(img,size,3)
plt.imshow(img)

# %%


def getImageRatio(image, ply):
    # Image size
    w, h = image.size

    # ply cx, cy
    # vert_ = ply['vertex']
    cx_w = ply['vertex']['cx'].max() + 1    # 256.0
    cy_h = ply['vertex']['cy'].max() + 1    # 192.0

    # resizing ratio
    w_r = w/cx_w
    h_r = h/cy_h

    return (w_r, h_r)


def _kps1d_to_2d(kps1d):
    kps = copy.deepcopy(kps1d)

    kps_num = divmod(len(kps), 3)[0]

    kps_2d_ls = [[kps[kp_i*3+0], kps[kp_i*3+1]]
                 for kp_i in range(kps_num) if kps[kp_i*3+2]]  # List
    kps_2d_arr = np.array(kps_2d_ls)

    return kps_2d_arr


def _kps_downscale(kps_arr_row, resize_r):  # ver.02 ***
    kps_arr = copy.deepcopy(kps_arr_row)
    w_r, h_r = resize_r

    kps_arr[:, 0] = kps_arr[:, 0]/w_r
    kps_arr[:, 1] = kps_arr[:, 1]/h_r

    return (kps_arr)


def get_proj_depth(ply):
    vert_ = ply['vertex']
    x_loc = vert_['cx'].max() + 1    # 256.0
    y_loc = vert_['cy'].max() + 1    # 192.0

    # proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32)
    proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32)

    for i in range(vert_.count):
        proj_depth[int(vert_['cx'][i]), int(
            vert_['cy'][i])] = vert_['depth'][i]

    return (proj_depth)

# %%


with open(output_est, 'r') as f:
    kps_val_json = json.load(f)

with open(valid_json_file, 'r') as f:
    val_coco_json = json.load(f)


imgid2img = {elem_['id']: str(elem_['id']).rjust(
    6, '0')+'.jpg' for elem_ in val_coco_json['images']}
imgid2ply = {elem_['id']: str(elem_['id']).rjust(
    6, '0')+'.ply' for elem_ in val_coco_json['images']}
imgid2kploc = {i: elem_['image_id'] for i, elem_ in enumerate(kps_val_json)}

# %%
img_values_list = list(imgid2img.values())
ply_values_list = list(imgid2ply.values())
# %%
missing_rows = df[df['name'].isnull()]

# %%
degree = 0.24
df = pd.DataFrame(index=range(len(img_values_list)),
                  columns=['name', 'width', 'height'])
for id_ in range(len(img_values_list)):
    try:
        image = Image.open(os.path.join(img_path, img_values_list[id_]))
        ply = PlyData.read(os.path.join(ply_path, ply_values_list[id_]))
        w_r, h_r = getImageRatio(image, ply)


#         ann_ = kps_val_json[imgid2kploc[id_]]
        ann_ = kps_val_json[id_]
#         print(id_, img_values_list[id_])
        kps1d = ann_['keypoints']

        kps_arr = _kps1d_to_2d(kps1d)  # 1D -> 2D
        kps_arr = _kps_downscale(kps_arr, (w_r, h_r))  # Down-Sizing
        # = {kpsID: [cx, cy]}
        kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)}

        ply_dpt = get_proj_depth(ply)

        a = list(map(int, kps_arr[0]))
        b = list(map(int, kps_arr[1]))
        c = list(map(int, kps_arr[2]))
        d = list(map(int, kps_arr[3]))
        # 변환한 keypoint점에 대응하는 ply값
        ply_a = ply_dpt[a[0], a[1]]
        ply_b = ply_dpt[b[0], b[1]]
        ply_c = ply_dpt[c[0], c[1]]
        ply_d = ply_dpt[d[0], d[1]]

        rad = np.deg2rad(degree)
        # a-c 점사이 거리 계산, heigth
        distance_a_to_c = np.sqrt(np.square(a[0]-c[0])+np.square(a[1]-c[1]))
        rad_a_to_c = distance_a_to_c*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_a_to_c = np.square(
            [ply_a*np.sin(rad_a_to_c), ply_c - ply_a*np.cos(rad_a_to_c)]).sum()
        size_a_to_c = np.sqrt(ss_a_to_c)
        height = size_a_to_c*1000
        # b-c 점사이 거리 계산, width
        distance_b_to_d = np.sqrt(np.square(b[0]-d[0])+np.square(b[1]-d[1]))
        rad_b_to_d = distance_b_to_d*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_b_to_d = np.square(
            [ply_b*np.sin(rad_b_to_d), ply_d - ply_b*np.cos(rad_b_to_d)]).sum()
        size_b_to_d = np.sqrt(ss_b_to_d)
        width = size_b_to_d*1000

        df.iloc[id_] = [list(imgid2img.values())[id_], width, height]
    except:
        '{0} has error'.format(img_values_list[id_])


# text 함수
# def Text(img,size,id):
#     cv2.putText(img, '{0}cm'.format(round(size['length'][id]['height']/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
#     cv2.putText(img, '{0}cm'.format(round(size['length'][id]['width']/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
#     plt.imshow(img)
#     return(img)


# %%
img, img_id, line_k = keypoint(3, output_est, img_path)
img = DrawLine(img, line_k)
img = Text(img, df, 0)
plt.imshow(img)

# %%


# %%

# %%
