import cv2
import numpy as np

from convert_json.utils.data_utils import get_affine_transform, normalize, get_gt_class_keypoints_dict
from convert_json.utils.infer_utils import get_final_preds, transform_preds, _coco_keypoint_results_all_category_kernel
from convert_json.utils.nms import oks_nms


# Data Settings
IMAGE_SIZE = [288, 384]
IMG_MEAN, IMG_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
C, S = np.array([959.5, 719.5]), np.array(
    [11.368751, 15.158334])  # center, scale

# post-processing of prediction setiings
target_weight = np.load('target_weight.npy')  # FIXED ???
gt_class_keypoints_dict = get_gt_class_keypoints_dict()  # fixed dictionary
heatmap_height = 96  # = config.MODEL.HEATMAP_SIZE[1]
heatmap_width = 72  # \ config.MODEL.HEATMAP_SIZE[0]

num_samples = 1
NUM_JOINTS = 294
IN_VIS_THRE = 0.2
OKS_THRE = 0.9


def pre(image_file_path: str, task_id: str):
    tflite_file = 'test_hrnet.tflite'
    print('tflite_file : ', tflite_file)
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    # Get input and output details(including the shape)
    input_details = interpreter.get_input_details()

    # Data load and transform
    data_numpy = cv2.imread(
        image_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    c, s, r = C, S, 0
    trans = get_affine_transform(c, s, r, IMAGE_SIZE)
    input_data = cv2.warpAffine(data_numpy, trans,
                                (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
                                flags=cv2.INTER_LINEAR)
    input_data = normalize(input_data, IMG_MEAN, IMG_STD)
    input_data = np.array(input_data, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0).transpose(0, 3, 1, 2)
    np.save(task_id + '_TEST_input_data', input_data)

    return {'task_id': task_id, 'index': input_details[0]['index'], 'input_data': input_data.tolist()}
