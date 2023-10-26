import numpy as np
import os
import json
import re
import math

# image dictionary 만드는 함수
def make_img(jsonfile_,ind_):
    new_images = {'coco_url': '',
    'date_captured': '',
    # json에서 숫자 출력 + josn 문자 추가 -> str로 나옴
    'file_name': '{0}'.format(jsonfile_['images'][ind_]['file_name']),
    'flickr_url': '',
    # jpg에서 숫자만 추출 -> int로 나옴
    'id': jsonfile_['images'][ind_]['id'],
    'license': 0,
    #width와 height 따로 입력
    'width': jsonfile_['images'][ind_]['width'],
    'height': jsonfile_['images'][ind_]['height']}
    
    return(new_images)
    
def make_cat(js):
    new_cat = {'id': 1,
    'name': 'strawberry',
    'supercategory': 'strawberry',
    'keypoints': ['1','2','3','4'],
    'skeletion': []}
    
    return(new_cat)
    
def make_keypoints(jsonfile_, ind_):
         
    #list flatten
    ls_2d = sum(jsonfile_['annotations'][ind_]['segmentation'],[])    
    #2d -> 3d ( 1 추가)
    ls_3d = [[round(ls_2d[i*2+0],2), round(ls_2d[i*2+1],2), 2] for i in range(4)]
    #3d to flatten
    ls = sum(ls_3d,[])
    
    null_key = [np.repeat(0.0,882)][0]
    
    for i in range(len(ls)):
        null_key[i] = ls[i]
    return(list(null_key))

def make_anno(jsonfile_, key_point, ind_):
    new_anno = {'area': jsonfile_['annotations'][ind_]['area'],
    'bbox': jsonfile_['annotations'][ind_]['bbox'],
    'category_id': 1,
    'id': jsonfile_['images'][ind_]['id'],
    'pair_id': jsonfile_['images'][ind_]['id'],
    'image_id': jsonfile_['annotations'][ind_]['image_id'],
    'iscrowd': 0,
    'style': 1,
    'num_keypoints': 4,
    'keypoints': key_point}
    
    return(new_anno)
    
def make_cat(jsonfile_):
    new_cat = {'id': 1,
    'name': 'strawberry',
    'supercategory': 'strawberry',
    'keypoints': jsonfile_['categories'][0]['keypoints'],
    'skeletion': []}
    
    return(new_cat)