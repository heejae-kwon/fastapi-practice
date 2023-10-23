# %%
import numpy as np
import os
import json
import re
import cv2

# %%
from os import path

# %% [markdown]
# # 데이터 압축 해제 후 파일명까지 변경한 이후 과정

# %%
dir = '/mnt/nas4/hjk/hrnet/data/strawberry/validation'

# %%
with open(os.path.join(dir, 'train-coco_style_template.json'), 'r') as f:
    json_template = json.load(f)

# %%
image_dir = os.listdir(dir+'/image')
filtered_list = [file for file in image_dir if file != '.ipynb_checkpoints']
numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]
sorted_numbers = sorted(numbers)
sorted_image_list = ['{0}'.format(sorted_number).rjust(
    6, '0')+'.jpg' for sorted_number in sorted_numbers]

# %% [markdown]
# - 딸기 이미지 image 카테고리 만드는 함수
# - file image와 file name을 이미지 카테고리에서 받아옴
# - width와 height는 이미지 shape로 설정

# %%


def make_img(dir, sorted_numbers, sorted_image_list, ind_):
    #     filtered_list = [file for file in image_dir if file != '.ipynb_checkpoints']
    #     numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]
    #     sorted_numbers = sorted(numbers)
    #     sorted_image_list = ['{0}'.format(sorted_number).rjust(6,'0')+'.jpg' for sorted_number in sorted_numbers]
    #     image = os.path.join(dir,'image',filtered_list[0])

    new_images = {'coco_url': '',
                  'date_captured': '',
                  # json에서 숫자 출력 + json 문자 추가 -> str로 나옴
                  'file_name': sorted_image_list[ind_],
                  'flickr_url': '',
                  # jpg에서 숫자만 추출 -> int로 나옴
                  'id': sorted_numbers[ind_],
                  'license': 0,
                  # width와 height 따로 입력
                  'width': cv2.imread(os.path.join(dir, 'image', sorted_image_list[ind_])).shape[1],
                  'height': cv2.imread(os.path.join(dir, 'image', sorted_image_list[ind_])).shape[0]}

    return (new_images)

# %% [markdown]
# - 딸기 아노테이션 만드는 함수
# - validation이기 때문에 0으로 채워줌

# %% [markdown]
# - 그런데 0으로 채워줬더니 인식을 못함
# - 랜덤값 12개를 넣어주자

# %%


def make_anno(sorted_numbers, ind_):
    null_key = list([np.repeat(0.0, 882)][0])
    for i in range(12):
        null_key[i] = [283.0, 191.0, 2.0, 284.0, 794.0,
                       2.0, 582.0, 793.0, 2.0, 581.0, 186.0, 2.0][i]
    new_anno = {'area': 33214,
                'bbox': [0, 0,
                         cv2.imread(os.path.join(
                             dir, 'image', sorted_image_list[ind_])).shape[1],
                         cv2.imread(os.path.join(
                             dir, 'image', sorted_image_list[ind_])).shape[0]
                         ],
                'category_id': 1,
                'id': sorted_numbers[ind_],
                'pair_id': sorted_numbers[ind_],
                'image_id': sorted_numbers[ind_],
                'iscrowd': 0,
                'style': 1,
                'num_keypoints': 4,
                'keypoints': null_key}

    return (new_anno)

# %% [markdown]
# - 마지막 카테고리 만드는 함수

# %%


def make_cat(ind_):
    ls = []
    for i in range(1, 295):
        ls.append('{}'.format(i))

    new_cat = {'id': 1,
               'name': 'strawberry',
               'supercategory': 'strawberry',
               'keypoints': ls,
               'skeletion': []}

    return (new_cat)

# %% [markdown]
# # 함수들을 이용해 coco_style_json 생성


# %%
with open(os.path.join(dir, 'train-coco_style_template.json'), 'r') as f:
    json_template = json.load(f)

# %%
for i in range(len(sorted_image_list)):
    json_template['images'].append(
        make_img(dir, sorted_numbers, sorted_image_list, i))
    json_template['annotations'].append(make_anno(sorted_numbers, i))

# %%
del (json_template['images'][0])
del (json_template['annotations'][0])

# %%
if path.isfile(os.path.join(dir, 'val-coco_style.json')):
    os.remove(os.path.join(dir, 'val-coco_style.json'))

# %%
with open(os.path.join(dir, 'val-coco_style.json'), 'w') as f:
    json.dump(json_template, f)
