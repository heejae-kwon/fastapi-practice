# %%
import time
import numpy as np
import os
import argparse
import zipfile
import shutil
import json
import re
import math
import pandas as pd

from os import path
from utils.make_template_segmentation import make_img, make_cat, make_keypoints, make_anno
from labelme2coco import get_coco_from_labelme_folder, save_json

# %%
## 데이터 디렉토리 경로 지정
data_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/strawberry/train'
valid_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/strawberry/validation'

# %%
## weight와 height가 있는 excel 불러오기 
M = pd.read_excel(os.path.join(data_dir, 'M_size.xlsx'),sheet_name=0)[1:]
L = pd.read_excel(os.path.join(data_dir, 'L_size.xlsx'),sheet_name=0)[1:]
L2 = pd.read_excel(os.path.join(data_dir, '2L_size.xlsx'),sheet_name=0)[1:]
## 컬럼 명명 
M.columns = ['order','number','width','height','weight','day','size','data']
L.columns = ['order','number','width','height','weight','day','size','data']
L2.columns = ['order','number','width','height','weight','day','size','data']
## 불필요한 컬럼 삭제
M.drop(['day','data'],axis=1, inplace=True)
L.drop(['day','data'],axis=1, inplace=True)
L2.drop(['day','data'],axis=1, inplace=True)
## NA 삭제
M.dropna(inplace=True)
L.dropna(inplace=True)
L2.dropna(inplace=True)

# %%
## 순서를 맞춰주기 위해서 하나의 데이터프레임 생성
size = pd.concat([M,L,L2], ignore_index=True)
size.reset_index(inplace=True)

## 순서를 맞춘 후 재분할
M = size[:302]
L = size[302:645]
L2 = size[645:]

## 갹 그룹별로 valid 샘플링
M_valid = size[:302].sample(frac=0.3,random_state=22).sort_values('order')
L_valid = size[302:645].sample(frac=0.3,random_state=22).sort_values('order')
L2_valid = size[645:].sample(frac=0.3,random_state=22).sort_values('order')

## valid set 
size_valid = pd.concat([M_valid, L_valid, L2_valid])

## train set 생성
M_train = M[~M.index.isin(M_valid.index)]
L_train = L[~L.index.isin(L_valid.index)]
L2_train = L2[~L2.index.isin(L2_valid.index)]

## 나는 M,L,2L train을 합친것을 train으로해서 돌리고, M,L,2L valida 합친것으로 validation을 돌리면 됌
size_train = pd.concat([M_train,L_train,L2_train])

# %%
img_dir = os.listdir(os.path.join(valid_dir,'image'))

# %%
numbers = [int(re.findall(r'\d+', file)[0]) for file in img_dir]

# %%
file_list = img_dir
filtered_list = [file for file in file_list if file != '.ipynb_checkpoints']

numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]
# sorted_numbers = sorted(numbers)

# %%
move_img_list = [20, 172, 232, 282, 297,348, 359, 381, 475, 510, 636, 640, 641, 643,732, 783, 821, 848, 849, 864, 874, 884]
# len(move_img_list)

# %%
import shutil

# %%
valid_dir

# %%
# validation에서 안쓰는 파일 옮기기 image 옮기는 것
for i, file_name in enumerate(filtered_list):
    for j, move_file_name in enumerate(move_img_list):
        if numbers[i] == move_file_name:
            src_path = os.path.join(valid_dir,'image',file_name)
            dst_path = os.path.join(valid_dir,'img_archieve',file_name)
#             print(src_path,dst_path)
            shutil.move(src_path, dst_path)
    

# %%


# %%
# annotation 옮기는 것
anno_dir = os.listdir(os.path.join(valid_dir, 'annos'))
file_list = anno_dir
filtered_list = [file for file in file_list if file != '.ipynb_checkpoints']

numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]

for i, file_name in enumerate(filtered_list):
    for j, move_file_name in enumerate(move_img_list):
        if numbers[i] == move_file_name:
            src_path = os.path.join(valid_dir,'annos',file_name)
            dst_path = os.path.join(valid_dir,'annos_archieve',file_name)
#             print(src_path,dst_path)
            shutil.move(src_path, dst_path)

# %%


# %%
# ply 옮기는 것
ply_dir = os.listdir(os.path.join(valid_dir, 'ply'))
file_list = ply_dir
filtered_list = [file for file in file_list if file != '.ipynb_checkpoints']

numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]

for i, file_name in enumerate(filtered_list):
    for j, move_file_name in enumerate(move_img_list):
        if numbers[i] == move_file_name:
            src_path = os.path.join(valid_dir,'ply',file_name)
            dst_path = os.path.join(valid_dir,'ply_archieve',file_name)
#             print(src_path,dst_path)
            shutil.move(src_path, dst_path)

# %%


# %%
# train에서 validation으로 22개의 새로운 파일 옮기기
train_image_dir = os.path.join(data_dir, 'image')
train_annos_dir = os.path.join(data_dir, 'annos')
train_ply_dir = os.path.join(data_dir, 'ply')

image = os.listdir(train_image_dir)
annos= os.listdir(train_annos_dir)
ply = os.listdir(train_ply_dir)

image_filtered = [file for file in image if file != '.ipynb_checkpoints']
annos_filtered = [file for file in annos if file != '.ipynb_checkpoints']
ply_filtered = [file for file in ply if file != '.ipynb_checkpoints']



# %%
import random
random.seed(22)
random_selection = random.sample(image_filtered, k=22)

# %%
valid_dir

# %%
# image 위치 변경
random.seed(22)
random_selection = random.sample(image_filtered, k=22)

for i, file_name in enumerate(random_selection):
    shutil.move( os.path.join(train_image_dir, file_name), os.path.join(valid_dir, 'image', file_name ))
#     print(os.path.join(valid_dir,'image', file_name))

# %%
random.seed(22)
random_selection = random.sample(annos_filtered, k=22)

for i, file_name in enumerate(random_selection):
    shutil.move( os.path.join(train_annos_dir, file_name), os.path.join(valid_dir, 'annos', file_name ))

# %%
random.seed(22)
random_selection = random.sample(ply_filtered, k=22)

for i, file_name in enumerate(random_selection):
    shutil.move( os.path.join(train_ply_dir, file_name), os.path.join(valid_dir, 'ply', file_name ))

# %%
img_dir = os.listdir(os.path.join(valid_dir,'image'))
annos_dir = os.listdir(os.path.join(valid_dir,'annos'))
ply_dir = os.listdir(os.path.join(valid_dir,'ply'))

print(len(img_dir), len(annos_dir), len(ply_dir))

# %%
img_dir = os.listdir(os.path.join(valid_dir,'image'))
annos_dir = os.listdir(os.path.join(valid_dir,'annos'))
ply_dir = os.listdir(os.path.join(valid_dir,'ply'))

# %%
len(img_dir)

# %%
len(annos_dir)

# %%
len(ply_dir)

# %% [markdown]
# ## valid coco style json

# %%
dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/strawberry/validation'

# %%
if os.path.exists(os.path.join(dir, 'dataset.json')):
    os.remove(os.path.join(dir, 'dataset.json'))    
if os.path.exists(os.path.join(dir, 'val_coco_style.json')):
        os.remove(os.path.join(dir, 'val_coco_style.json'))

# %%
coco_dataset = get_coco_from_labelme_folder(os.path.join(dir,'annos'))
save_json(coco_dataset.json, os.path.join(dir,'dataset.json'))

# %%
file_list = os.listdir(os.path.join(dir,'image'))
filtered_list = [file for file in file_list if file != '.ipynb_checkpoints']
numbers = [int(re.search(r'\d+', file).group()) for file in filtered_list]
sorted_numbers = sorted(numbers)

# %%
with open(os.path.join(dir,'dataset.json'),'r') as f:
    dataset = json.load(f)
    
for i in range(len(dataset['images'])):
    dataset['images'][i]['id'] = sorted_numbers[i]+1
    dataset['annotations'][i]['image_id'] = sorted_numbers[i]+1
    dataset['annotations'][i]['id'] = sorted_numbers[i]+1

with open(os.path.join(dir,'dataset.json'),'w') as f:
    json.dump(dataset, f)    

# %%
with open(os.path.join(dir,'train-coco_style_template.json')  ,'r') as f:
    json_template = json.load(f)
with open(os.path.join(dir,'dataset.json'),'r') as f:
    dataset = json.load(f)

for i in range(len(dataset['images'])):
    json_template['images'].append(make_img(dataset,i))
    json_template['annotations'].append(make_anno(dataset,make_keypoints(dataset,i),i))

del(json_template['images'][0])
del(json_template['annotations'][0])

with open(os.path.join(dir,'val-coco_style.json'),'w') as f:
        json.dump(json_template, f)

# %% [markdown]
# ## 점 찍기

# %%
size = pd.concat([M,L,L2], ignore_index=True)
size.reset_index(inplace=True)

# %%
for i, index_ in enumerate(size.shape[0]):
    for j, number in numbers:
        if 
    

# %%
# 해당하는 열만 가져오기
selected_rows = size[size['index'].isin(numbers)]

# %%
import os
import os.path as osp
import copy
from tqdm import tqdm

import json
import math
import numpy as np
import pandas as pd
from PIL import Image
import imageio as iio
import cv2
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud
from plyfile import PlyData

import matplotlib.pyplot as plt

from utils.size_info import getSizingPts
from utils.text_position_info import position_info

# %%
# 점 찍는 함수
def keypoint(id,output,image_dir):
    with open(output,'r') as f:
        keypoint_json = json.load(f)
        
    image_id = keypoint_json[id]['image_id']
    keypoints = keypoint_json[id]['keypoints']
    
    if image_id<10:    
        img = cv2.imread('{0}/00000{1}.jpg'.format(image_dir,image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    elif image_id<100:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir,image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    elif image_id<1000:
        img = cv2.imread('{0}/000{1}.jpg'.format(image_dir,image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    elif image_id<10000:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir,image_id))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    kpt = keypoints[:12]
    line_k = []
    
    for k in range(0,len(kpt),3) :
        x = int(kpt[k])
        y = int(kpt[k+1])
        line_k.extend([x,y])
        cv2.circle(img, (x,y), 4, (0,255,0),-1)

        
    return(img, image_id, line_k)

# 선 그리는 함수
def DrawLine(img, line_k):
    cv2.line(img, (line_k[0],line_k[1]), (line_k[4],line_k[5]), (0,255,0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (line_k[2],line_k[3]), (line_k[6],line_k[7]), (0,255,0), thickness=6, lineType=cv2.LINE_AA)
    return(img)
    
# text 함수
def Text(img,size,id):
    cv2.putText(img, '{0}cm'.format(round(size['length'][id]['height']/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    cv2.putText(img, '{0}cm'.format(round(size['length'][id]['width']/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    plt.imshow(img)
    return(img)


# %%
output_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/output/straw_try/pose_hrnet/keypoint_strawberry'
output_est=os.path.join(output_dir,os.listdir(output_dir)[-1],'results/keypoints_validation_results_0.json' )

valid_dir ='/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/strawberry/validation'
valid_json_file = os.path.join(valid_dir,'val-coco_style.json')
output=os.path.join(valid_dir,'keypoints_validation_results_0.json')

img_path = os.path.join(valid_dir,'image')
ply_path = os.path.join(valid_dir,'ply')

# %%
img, img_id, line_k = keypoint(15,output_est, img_path)
img = DrawLine(img,line_k)
# img = Text(img,size,3)
plt.imshow(img)

# %%
def getImageRatio(image,ply): 
    # Image size 
    w, h = image.size

    # ply cx, cy 
    # vert_ = ply['vertex']
    cx_w = ply['vertex']['cx'].max() + 1    # 256.0 
    cy_h = ply['vertex']['cy'].max() + 1    # 192.0 

    # resizing ratio 
    w_r = w/cx_w
    h_r = h/cy_h

    return(w_r, h_r)

def _kps1d_to_2d(kps1d):
    kps = copy.deepcopy(kps1d)

    kps_num = divmod(len(kps), 3)[0]

    kps_2d_ls = [[kps[kp_i*3+0], kps[kp_i*3+1]] for kp_i in range(kps_num) if kps[kp_i*3+2]] # List
    kps_2d_arr = np.array(kps_2d_ls)

    return kps_2d_arr

def _kps_downscale(kps_arr_row, resize_r): # ver.02 ***    
    kps_arr = copy.deepcopy(kps_arr_row)
    w_r, h_r = resize_r 
    
    kps_arr[:, 0] = kps_arr[:, 0]/w_r
    kps_arr[:, 1] = kps_arr[:, 1]/h_r

    return(kps_arr)

def get_proj_depth(ply): 
    vert_ = ply['vertex']
    x_loc = vert_['cx'].max() + 1    # 256.0 
    y_loc = vert_['cy'].max() + 1    # 192.0 

    # proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32) 
    proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32) 

    for i in range(vert_.count): 
        proj_depth[int(vert_['cx'][i]), int(vert_['cy'][i])] = vert_['depth'][i] 

    return(proj_depth) 

# %%
with open(output_est, 'r') as f:
    kps_val_json = json.load(f)

with open(valid_json_file, 'r') as f:
    val_coco_json = json.load(f)

# %%
def size_estimator(degree, xlsx_=False):
    # real_size 실제 거리
#     real_size = pd.read_excel(os.path.join(valid_dir, xlsx_),sheet_name=1)[1:]
#     real_size = real_size.drop(['Unnamed: 0'],axis=1)
#     real_size.columns = ['width','height','weight']


    # size_estimate 거리 예측
    imgid2img = {elem_['id']: str(elem_['id']).rjust(6,'0')+'.jpg' for elem_ in val_coco_json['images']}
    imgid2ply = {elem_['id']: str(elem_['id']).rjust(6,'0')+'.ply' for elem_ in val_coco_json['images']}

    imgid2kploc = {elem_['image_id']: i for i, elem_ in enumerate(kps_val_json)}

    # degree 및 length와 같은 정보를 담을 빈 리스트 생성
    new_id_size_estimator = []
    new_length_size_estimator = []

    for id_ in range(imgid_from+1, imgid_to+1):
        try:

            image = Image.open(os.path.join(img_path, imgid2img[id_]))
            ply = PlyData.read(os.path.join(ply_path, imgid2ply[id_]))
            w_r, h_r = getImageRatio(image,ply)

            ann_ = kps_val_json[imgid2kploc[id_]]
            kps1d = ann_['keypoints']

            kps_arr = _kps1d_to_2d(kps1d) # 1D -> 2D 
            kps_arr = _kps_downscale(kps_arr, (w_r, h_r)) # Down-Sizing 
            kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)} # = {kpsID: [cx, cy]} 

            ply_dpt = get_proj_depth(ply) 

            a = list(map(int,kps_arr[0]))
            b = list(map(int,kps_arr[1]))
            c = list(map(int,kps_arr[2]))
            d = list(map(int,kps_arr[3]))
        #### 변환한 keypoint점에 대응하는 ply값
            ply_a = ply_dpt[a[0], a[1]]
            ply_b = ply_dpt[b[0], b[1]]
            ply_c = ply_dpt[c[0], c[1]]
            ply_d = ply_dpt[d[0], d[1]]

            rad = np.deg2rad(degree)
        ## a-c 점사이 거리 계산, heigth
            distance_a_to_c = np.sqrt(np.square(a[0]-c[0])+np.square(a[1]-c[1]))
            rad_a_to_c = distance_a_to_c*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

            ss_a_to_c = np.square([ply_a*np.sin(rad_a_to_c), ply_c - ply_a*np.cos(rad_a_to_c)]).sum()
            size_a_to_c = np.sqrt(ss_a_to_c)
            height = size_a_to_c*1000
            ## b-c 점사이 거리 계산, width
            distance_b_to_d = np.sqrt(np.square(b[0]-d[0])+np.square(b[1]-d[1]))
            rad_b_to_d = distance_b_to_d*rad

            # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
            # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

            ss_b_to_d = np.square([ply_b*np.sin(rad_b_to_d), ply_d - ply_b*np.cos(rad_b_to_d)]).sum()
            size_b_to_d = np.sqrt(ss_b_to_d)
            width = size_b_to_d*1000

            # 이미지에 대한 정보 및 설정한 각도 추가
            new_id_size_estimator.append({
                'id_{0}'.format(id_) : degree
            })
            # length 추가
            new_length_size_estimator.append({
                'width' : round(width,1),
                'height' : round(height,1),
#                 'real_width' : real_size['width'][id_],
#                 'real_height' : real_size['height'][id_]
            })
        except:
            new_id_size_estimator.append({
                'id_{0}'.format(id_) : degree
            })
            new_length_size_estimator.append({
                None
            })
    size_estimate = {
    'id' : new_id_size_estimator,
    'length' : new_length_size_estimator
    }
    return(size_estimate)

# %%


# %%
id_ = 260
image = Image.open(os.path.join(img_path, img_values_list[id_]))
ply = PlyData.read(os.path.join(ply_path, ply_values_list[id_]))
w_r, h_r = getImageRatio(image,ply)


#     ann_ = kps_val_json[imgid2kploc[id_]]
ann_ = kps_val_json[id_]
print(id_, img_values_list[id_])
kps1d = ann_['keypoints']

kps_arr = _kps1d_to_2d(kps1d) # 1D -> 2D 
kps_arr = _kps_downscale(kps_arr, (w_r, h_r)) # Down-Sizing 
kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)} # = {kpsID: [cx, cy]} 

ply_dpt = get_proj_depth(ply) 

a = list(map(int,kps_arr[0]))
b = list(map(int,kps_arr[1]))
c = list(map(int,kps_arr[2]))
d = list(map(int,kps_arr[3]))
#### 변환한 keypoint점에 대응하는 ply값
ply_a = ply_dpt[a[0], a[1]]
ply_b = ply_dpt[b[0], b[1]]
ply_c = ply_dpt[c[0], c[1]]
ply_d = ply_dpt[d[0], d[1]]

rad = np.deg2rad(degree)
## a-c 점사이 거리 계산, heigth
distance_a_to_c = np.sqrt(np.square(a[0]-c[0])+np.square(a[1]-c[1]))
rad_a_to_c = distance_a_to_c*rad

# 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
# ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

ss_a_to_c = np.square([ply_a*np.sin(rad_a_to_c), ply_c - ply_a*np.cos(rad_a_to_c)]).sum()
size_a_to_c = np.sqrt(ss_a_to_c)
height = size_a_to_c*1000
## b-c 점사이 거리 계산, width
distance_b_to_d = np.sqrt(np.square(b[0]-d[0])+np.square(b[1]-d[1]))
rad_b_to_d = distance_b_to_d*rad

# 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
# ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

ss_b_to_d = np.square([ply_b*np.sin(rad_b_to_d), ply_d - ply_b*np.cos(rad_b_to_d)]).sum()
size_b_to_d = np.sqrt(ss_b_to_d)
width = size_b_to_d*1000

height,width
# df.iloc[id_] = [list(imgid2img.values())[id_], weight, height]

# %%
degree = 0.24
df = pd.DataFrame(index=range(len(img_values_list)), columns=['name','width','height'])
for id_ in range(len(img_values_list)):
    try:
        image = Image.open(os.path.join(img_path, img_values_list[id_]))
        ply = PlyData.read(os.path.join(ply_path, ply_values_list[id_]))
        w_r, h_r = getImageRatio(image,ply)


#         ann_ = kps_val_json[imgid2kploc[id_]]
        ann_ = kps_val_json[id_]
#         print(id_, img_values_list[id_])
        kps1d = ann_['keypoints']

        kps_arr = _kps1d_to_2d(kps1d) # 1D -> 2D 
        kps_arr = _kps_downscale(kps_arr, (w_r, h_r)) # Down-Sizing 
        kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)} # = {kpsID: [cx, cy]} 

        ply_dpt = get_proj_depth(ply) 

        a = list(map(int,kps_arr[0]))
        b = list(map(int,kps_arr[1]))
        c = list(map(int,kps_arr[2]))
        d = list(map(int,kps_arr[3]))
        #### 변환한 keypoint점에 대응하는 ply값
        ply_a = ply_dpt[a[0], a[1]]
        ply_b = ply_dpt[b[0], b[1]]
        ply_c = ply_dpt[c[0], c[1]]
        ply_d = ply_dpt[d[0], d[1]]

        rad = np.deg2rad(degree)
        ## a-c 점사이 거리 계산, heigth
        distance_a_to_c = np.sqrt(np.square(a[0]-c[0])+np.square(a[1]-c[1]))
        rad_a_to_c = distance_a_to_c*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_a_to_c = np.square([ply_a*np.sin(rad_a_to_c), ply_c - ply_a*np.cos(rad_a_to_c)]).sum()
        size_a_to_c = np.sqrt(ss_a_to_c)
        height = size_a_to_c*1000
        ## b-c 점사이 거리 계산, width
        distance_b_to_d = np.sqrt(np.square(b[0]-d[0])+np.square(b[1]-d[1]))
        rad_b_to_d = distance_b_to_d*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_b_to_d = np.square([ply_b*np.sin(rad_b_to_d), ply_d - ply_b*np.cos(rad_b_to_d)]).sum()
        size_b_to_d = np.sqrt(ss_b_to_d)
        width = size_b_to_d*1000

        df.iloc[id_] = [list(imgid2img.values())[id_], width, height]
    except:
        '{0} has error'.format(img_values_list[id_])

# %%
imgid2img = {elem_['id']: str(elem_['id']).rjust(6,'0')+'.jpg' for elem_ in val_coco_json['images']}
imgid2ply = {elem_['id']: str(elem_['id']).rjust(6,'0')+'.ply' for elem_ in val_coco_json['images']}
imgid2kploc = {i : elem_['image_id'] for i, elem_ in enumerate(kps_val_json)}

# %%
img_values_list = list(imgid2img.values())
ply_values_list = list(imgid2ply.values())

# %%
missing_rows = df[df['name'].isnull()]

# %%
degree = 0.24
df = pd.DataFrame(index=range(len(img_values_list)), columns=['name','width','height'])
for id_ in range(len(img_values_list)):
    try:
        image = Image.open(os.path.join(img_path, img_values_list[id_]))
        ply = PlyData.read(os.path.join(ply_path, ply_values_list[id_]))
        w_r, h_r = getImageRatio(image,ply)


#         ann_ = kps_val_json[imgid2kploc[id_]]
        ann_ = kps_val_json[id_]
#         print(id_, img_values_list[id_])
        kps1d = ann_['keypoints']

        kps_arr = _kps1d_to_2d(kps1d) # 1D -> 2D 
        kps_arr = _kps_downscale(kps_arr, (w_r, h_r)) # Down-Sizing 
        kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)} # = {kpsID: [cx, cy]} 

        ply_dpt = get_proj_depth(ply) 

        a = list(map(int,kps_arr[0]))
        b = list(map(int,kps_arr[1]))
        c = list(map(int,kps_arr[2]))
        d = list(map(int,kps_arr[3]))
        #### 변환한 keypoint점에 대응하는 ply값
        ply_a = ply_dpt[a[0], a[1]]
        ply_b = ply_dpt[b[0], b[1]]
        ply_c = ply_dpt[c[0], c[1]]
        ply_d = ply_dpt[d[0], d[1]]

        rad = np.deg2rad(degree)
        ## a-c 점사이 거리 계산, heigth
        distance_a_to_c = np.sqrt(np.square(a[0]-c[0])+np.square(a[1]-c[1]))
        rad_a_to_c = distance_a_to_c*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_a_to_c = np.square([ply_a*np.sin(rad_a_to_c), ply_c - ply_a*np.cos(rad_a_to_c)]).sum()
        size_a_to_c = np.sqrt(ss_a_to_c)
        height = size_a_to_c*1000
        ## b-c 점사이 거리 계산, width
        distance_b_to_d = np.sqrt(np.square(b[0]-d[0])+np.square(b[1]-d[1]))
        rad_b_to_d = distance_b_to_d*rad

        # 사이각에 대한 코사인 갑 계산 -> 사이거리 계산
        # ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()

        ss_b_to_d = np.square([ply_b*np.sin(rad_b_to_d), ply_d - ply_b*np.cos(rad_b_to_d)]).sum()
        size_b_to_d = np.sqrt(ss_b_to_d)
        width = size_b_to_d*1000

        df.iloc[id_] = [list(imgid2img.values())[id_], width, height]
    except:
        '{0} has error'.format(img_values_list[id_])

# %%
selected_rows.reset_index(inplace=True)
selected_rows.drop('level_0', axis=1, inplace=True)

# %%
df['width_차이'] = df['width'] - selected_rows['width']
df['height_차이'] = df['height'] - selected_rows['height']

# %%
df

# %%
df[['width','height']] = df[['width','height']].astype(float)
selected_rows[['width','height']] = selected_rows[['width','height']].astype(float)

# %%
# for i in range(len(df)):
#     df['width_차이'].iloc[i] = df['width'].iloc[i] - selected_rows['width'].iloc[i]
#     df['height_차이'].iloc[i] = df['height'].iloc[i] - selected_rows['height'].iloc[i]

# %%
for i in range(len(df)):
    try:
        df['index'].iloc[i] = int(df['name'].iloc[i].split('.')[0])
    except:
        pass

# %%
size_valid = selected_rows.rename(columns={'width': 'width_실제거리', 'height': 'height_실제거리'})

# %%
len(df)

# %%
int(df['name'][0].split('.')[0])

# %%
df['index'] = int(df['name'].iloc[i].split('.')[0])

# %%
for i in range(len(df)):
    try:
        df['index'].iloc[i] = int(df['name'][i].split('.')[0])
    except:
        print(i)
        pass

# %%
df

# %%
size_valid

# %%
merged_df = size_valid.merge(df, on='index')

# %%
merged_df.isnull()

# %%
size_valid['index'].sum()

# %%
df['index'].sum()

# %%
num = [file for file in img_dir if file != '.ipynb_checkpoints']

# %%
filtered_img_dir = [file for file in img_dir if file != '.ipynb_checkpoints'] 
num = [int(file.split('.')[0]) for file in filtered_img_dir]

# %%
sum(num)

# %%
size.iloc[846]

# %%
df[250:]

# %%
df[df['index']!=size_valid['index']]

# %%
false_rows = df[df['index'] != size_valid['index']]

# %%
merged_df[250:]

# %%
df.iloc[260] =  ['000845.jpg',54.105841,53.847915,-0.894159, -1.152085,845]

# %%
df.iloc[260][['name','width','height','width_차이','height_차이','index']] = ['000845.jpg',54.105841,53.847915,-0.894159, -1.152085,845]

# %%
df.iloc[260]

# %%
merged_df = size_valid.merge(df, on='index')

# %%
sum(merged_df.isnull())

# %%
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

# %%
def reg(width,height):
    weight = 1.2166*width + 0.4931*height - 41.0601
    return(weight)

# %%
reg(54.105841,53.847915)

# %%
for i in range(5):
    weight = reg(df.iloc[i]['width'],df.iloc[i]['height'])
    print(weight)

# %%
merged_df['weight_예측'] = 0

# %%
for i in range(len(merged_df)):
    try:
        weight = reg(merged_df.iloc[i]['width'],merged_df.iloc[i]['height'])
        merged_df['weight_예측'].iloc[i] = weight
    except:
        pass

# %%
merged_df['weight_차이'] = 0

# %%
for i in range(len(df)):
    merged_df['weight_차이'].iloc[i] = merged_df['weight_예측'].iloc[i] - size_valid['weight'].iloc[i]

# %%
merged_df

# %%
merged_df.to_excel('strawberry_estimation_vol2.xlsx')

# %%


# %%
# 점 찍는 함수
def keypoint(id,output,image_dir):
    with open(output,'r') as f:
        keypoint_json = json.load(f)
        
    image_id = keypoint_json[id]['image_id']
    keypoints = keypoint_json[id]['keypoints']
    
    if image_id<10:    
        img = cv2.imread('{0}/00000{1}.jpg'.format(image_dir,image_id))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    elif image_id<100:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir,image_id))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    elif image_id<1000:
        img = cv2.imread('{0}/000{1}.jpg'.format(image_dir,image_id))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    elif image_id<10000:
        img = cv2.imread('{0}/0000{1}.jpg'.format(image_dir,image_id))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    kpt = keypoints[:12]
    line_k = []
    
    for k in range(0,len(kpt),3) :
        x = int(kpt[k])
        y = int(kpt[k+1])
        line_k.extend([x,y])
        cv2.circle(img, (x,y), 15, (0,0,0),-1)

        
    return(img, image_id, line_k)

# 선 그리는 함수
def DrawLine(img, line_k):
    cv2.line(img, (line_k[0],line_k[1]), (line_k[4],line_k[5]), (0,255,0), thickness=6, lineType=cv2.LINE_AA)
    cv2.line(img, (line_k[2],line_k[3]), (line_k[6],line_k[7]), (0,255,0), thickness=6, lineType=cv2.LINE_AA)
    return(img)
    
# text 함수
# def Text(img,size,id):
#     cv2.putText(img, '{0}cm'.format(round(size['length'][id]['height']/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
#     cv2.putText(img, '{0}cm'.format(round(size['length'][id]['width']/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
#     plt.imshow(img)
#     return(img)

def Text(img, df, id):
    cv2.putText(img, '{0}cm'.format(round(df['height'].iloc[id]/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    cv2.putText(img, '{0}cm'.format(round(df['width'].iloc[id]/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    return(img)

# %%
img, img_id, line_k = keypoint(3,output_est, img_path)
img = DrawLine(img,line_k)
img = Text(img,merged_df,0)
plt.imshow(img)

# %%


# %%
for i in range(len(df)):
    try:
        img, img_id, line_k = keypoint(i,output_est, img_path)
        img = DrawLine(img,line_k)
        img = Text(img,merged_df,i)
        cv2.imwrite(os.path.join('/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/output_image_dir_vol2',merged_df['name'].iloc[i]),img)
    except:
        print('{0} has error'.format(merged_df['name'].iloc[i]))

# %%
import zipfile
import os

output_image_dir = '/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/output_image_dir_vol3'
# 압축할 파일들이 있는 폴더 경로

# 압축 파일 이름과 경로 설정
zip_filename = 'output.zip'
zip_filepath = os.path.join(output_image_dir, zip_filename)

# 압축할 파일 리스트 생성
file_list = os.listdir(output_image_dir)

# 압축 파일 생성 및 파일 추가
with zipfile.ZipFile(zip_filepath, 'w') as zipf:
    for file in file_list:
        file_path = os.path.join(output_image_dir, file)
        zipf.write(file_path, file)

print('압축이 완료되었습니다.')

# %%


# %%
df.iloc[32]

# %%
def Text(img, df, id):
    cv2.putText(img, '{0}cm'.format(round(df['height'].loc[id]/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    cv2.putText(img, '{0}cm'.format(round(df['width'].loc[id]/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
    return(img)

# %%
for i in range(len(df)):
    try:
        img, img_id, line_k = keypoint(i,output_est, img_path)
        img = DrawLine(img,line_k)
        img = Text(img,df,i)
        cv2.imwrite(os.path.join('/mnt/nas4/mhj/HRNet-for-Fashion-Landmark-Estimation.PyTorch/output_image_dir_vol3',df['name'].iloc[i]),img)
    except:
        print('{0} has error'.format(df['name'].iloc[i]))

# %%
img, img_id, line_k = keypoint(0,output_est, img_path)

# %%
img = DrawLine(img,line_k)
img = Text(img,df,2)

# %%
id = 0
cv2.putText(img, '{0}cm'.format(round(df['height'].iloc[id]/10,2)), (round((line_k[0]+line_k[4])/2),round((line_k[1]+line_k[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
cv2.putText(img, '{0}cm'.format(round(df['width'].iloc[id]/10,2)), (line_k[2],line_k[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0),2 )
plt.imshow(img)


