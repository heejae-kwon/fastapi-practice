import json
import re
import shutil
import cv2
import numpy as np

from pathlib import Path

dir = Path('strawberry/data/validation')


def init_and_rename_imgs(src_dir: Path):
    # 이미지 파일 초기화 및 이름 변경
    dest_dir = dir / "image"

    try:
        # 대상 디렉토리 경로

        # 디렉토리가 존재하는지 확인
        if not dest_dir.exists():
            print(f"디렉토리 '{dest_dir}'가 존재하지 않습니다.")
            return

        # 디렉토리 및 하위 내용 모두 삭제
        shutil.rmtree(dest_dir)

        print(f"디렉토리 '{dest_dir}'와 모든 하위 내용이 삭제되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

    for src_file in src_dir.glob('**/*'):
        if src_file.is_file():
            dest_file = dest_dir / src_file.relative_to(src_dir)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_file), str(dest_file))

    img_files = list(dest_dir.glob('*.jpg'))  # + list(dest_dir.glob('*.png'))
    for index, img_file in enumerate(img_files):
        extension = img_file.suffix
        new_file_name = f"{index:06d}{extension}"
        new_file_path = dest_dir / new_file_name
        img_file.rename(new_file_path)


def _make_img(sorted_numbers, sorted_image_list, ind_):
    img_path = dir / 'image' / sorted_image_list[ind_]
    img = cv2.imread(str(img_path))

    new_images = {
        'coco_url': '',
        'date_captured': '',
        'file_name': sorted_image_list[ind_],
        'flickr_url': '',
        'id': sorted_numbers[ind_],
        'license': 0,
        'width': img.shape[1],
        'height': img.shape[0]
    }

    return new_images


def _make_anno(sorted_numbers, sorted_image_list, ind_):
    null_key = list([np.repeat(0.0, 882)][0])
    for i in range(12):
        null_key[i] = [283.0, 191.0, 2.0, 284.0, 794.0,
                       2.0, 582.0, 793.0, 2.0, 581.0, 186.0, 2.0][i]

    image_path = Path(dir) / 'image' / sorted_image_list[ind_]
    image_shape = cv2.imread(str(image_path))

    new_anno = {
        'area': 33214,
        'bbox': [0, 0, image_shape.shape[1], image_shape.shape[0],
                 ],
        'category_id': 1,
        'id': sorted_numbers[ind_],
        'pair_id': sorted_numbers[ind_],
        'image_id': sorted_numbers[ind_],
        'iscrowd': 0,
        'style': 1,
        'num_keypoints': 4,
        'keypoints': null_key
    }

    return new_anno


def create_coco_json():

    # Load the JSON template
    with (dir / 'train-coco_style_template.json').open('r') as f:
        json_template = json.load(f)

    # List image files in the image directory
    image_dir = list((dir / 'image').glob('*.jpg'))
    filtered_list = [
        file for file in image_dir
    ]
    numbers = [int(re.search(r'\d+', file.name).group())
               for file in filtered_list]
    sorted_numbers = sorted(numbers)
    sorted_image_list = [
        f'{sorted_number:06d}.jpg' for sorted_number in sorted_numbers]

    # Load the JSON template again (not sure why you are loading it twice)
    with (dir / 'train-coco_style_template.json').open('r') as f:
        json_template = json.load(f)

    # Iterate over image files and create entries in JSON
    for i in range(len(sorted_image_list)):
        json_template['images'].append(
            _make_img(sorted_numbers, sorted_image_list, i))
        json_template['annotations'].append(
            _make_anno(sorted_numbers, sorted_image_list, i))

    # Remove the first entry in 'images' and 'annotations'
    del json_template['images'][0]
    del json_template['annotations'][0]

    # Remove the old 'val-coco_style.json' if it exists
    val_coco_json = dir / 'val-coco_style.json'
    if val_coco_json.is_file():
        val_coco_json.unlink()

    # Write the updated JSON to 'val-coco_style.json'
    with val_coco_json.open('w') as f:
        json.dump(json_template, f)

# Define _make_img and _make_anno functions as needed

# Example usage
