from .u2net import data_loader, u2net

from pathlib import Path
from torchvision import transforms
from PIL import ImageFile
from PIL import Image

import torch
import matplotlib.pyplot as plt
import io
import cv2
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True
device_config = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_config)
print('device :', device)


class _Rembg:
    def __init__(self, task_path: Path):
        self._task_path: Path = task_path

    def _preprocess(self, image: np.ndarray):
        label_3 = np.zeros(image.shape)
        label = np.zeros(label_3.shape[0:2])

        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        transform = transforms.Compose(
            [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
        )

        sample = transform(
            {"imidx": np.array([0]), "image": image, "label": label})

        result_img = sample['image'].numpy()[0]
        plt.imshow(result_img)
        plt.show()
        plt.savefig(str(self._task_path/Path("SAVE_FIG.jpg")))

        return sample

    def _norm_pred(self, d: torch.Tensor):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)

        return dn

    def _naive_cutout(self, img, mask):
        empty = Image.new("RGBA", (img.size), 0)  # 원본 이미지 size로 empty 생성

        # Image.composite : Image 합성 함수
        # 리사이즈 방법에는 크게 bilinear, bicubic, lanczos 이렇게 세 가지가 있습니다.
        # bilinear로 리사이즈 할 경우 용량이 더 작아지고 인코딩 속도도 빠르지만 흐릿한 느낌을 주는 반면,
        # lanczos 방식은 용량도 커지고 인코딩 속도도 느리지만 가장 선명한 화질을 보여줍니다.
        #  ->  Lanczos. 일반적으로 지원되는 알고리즘 중 가장 고품질의 이미지를 얻을 수 있다.
        #  - > Lanczos. 푸리에와 비슷한 아이디어로 처리되는듯한 sinc ??
        # bicubic은 용량, 속도, 선명함에서 중간 정도라고 보시면 될 듯 합니다.
        # Lanzos : 이미지 보간 BILINEAR 같은 method임
        cutout = Image.composite(
            img, empty, mask.resize(img.size, Image.LANCZOS))
        return cutout

    def remove_background(self, input_path, model, alpha_matting=False):
        print(f'REMOVE BG : {str(self._task_path)}')
        org_img = Image.open(input_path)
        f = np.fromfile(input_path)
        print('  org_img shape --> ', np.array(org_img).shape)
        plt.imshow(org_img)
        plt.title('Origin Image')
        plt.show()
        plt.savefig(str(self._task_path/Path("ORIGIN_IMAGE.jpg")))

        ##################################################
        ################# PREDICT ########################
        img = Image.open(io.BytesIO(f)).convert("RGB")

        sample = self._preprocess(np.array(img))
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs_test = torch.cuda.FloatTensor(
                    sample["image"].unsqueeze(0).cuda().float()
                )
            else:
                inputs_test = torch.FloatTensor(
                    sample["image"].unsqueeze(0).float())

            """
                        torch.onnx.export(model,               # 실행될 모델
                            # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                            inputs_test,
                            # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                            "u2net.onnx",
                            export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                            do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                            input_names=['input'],   # 모델의 입력값을 가리키는 이름
                            output_names=['output'],  # 모델의 출력값을 가리키는 이름
                            dynamic_axes={'input': {0: 'batch_size'},    # 가변적인 길이를 가진 차원
                                            'output': {0: 'batch_size'}})

            scripted_module = torch.jit.script(model)
            # Export lite interpreter version model (compatible with lite interpreter)
            scripted_module._save_for_lite_interpreter("u2net.ptl")
    """

            d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)

            pred = d1[:, 0, :, :]
            predict: torch.Tensor = self._norm_pred(pred)

            predict = predict.squeeze()
            predict_np = predict.cpu().detach().numpy()
    #         print('predict_np',predict_np*255)
            mask = Image.fromarray(predict_np * 255).convert("RGB")
            mask = mask.convert('L')
    #         predict_np[predict_np>0.01] = 1
    #         predict_np[predict_np<0.01] = 0
    #         mask = predict_np

            del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample
        #################################################

        print('The predicted size of the mask size --> ', np.array(mask).shape)

        plt.title('Predicted Mask')
        plt.imshow(mask, cmap='gray')
        plt.show()
        plt.savefig(str(self._task_path/Path("PREDICTED_MASK.jpg")))

        '''    if alpha_matting:
                try:
                    print(' alpha_matting! ')
                    cutout = alpha_matting_cutout(
                        img,
                        mask,
                        foreground_threshold=240,
                        background_threshold=10,
                        erode_structure_size=15,  # "-ae",
                        base_size=1000
                    )
                except Exception:
                    cutout = naive_cutout(img, mask)

            else:
        '''
        cutout = self._naive_cutout(img, mask)
        # cutout = naive_cutout(img.convert("RGBA"), mask.convert("RGBA"))

        bio = io.BytesIO()
        cutout.save(bio, "PNG")
        result = bio.getbuffer()
        result_img = Image.open(io.BytesIO(result)).convert("RGBA")

        plt.subplot(1, 4, 3)
        plt.imshow(mask.resize(img.size, Image.LANCZOS), cmap='gray')
        plt.title('Predicted Mask(resize)')

        print('  result_img shape :', np.array(result_img).shape)
        plt.subplot(1, 4, 4)
        plt.imshow(result_img)
        plt.title('Predicted Image')
        plt.show()

        return result_img


def run_rembg(
    task_folder_path: Path
):
    image_path = task_folder_path / Path('img_file.jpg')
    model_path = Path('rembg/model/rembg_UNETPmodel2021_11_25.pt')
    model_path = Path('rembg/model/rembg_UNETmodel2021_11_25.pt')
    # model_path = './rembg_UNETHUMANSEGmodel2021_11_25.pt'

    rembg = _Rembg(task_path=task_folder_path)

    print('model path:', model_path)

    if 'UNETP' in str(model_path):
        print('model : ', 'UNETP')
        model = u2net.U2NETP(3, 1)
    else:
        print('model : ', 'UNET')
        model = u2net.U2NET(3, 1)  # UNET and u2net_human_seg

    # model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())) )
    model.to(device)
    # Basically, the [GPU-Memory] allocated by the model -> 2000MB
    model.load_state_dict(torch.load(
        str(model_path.absolute()))['model_state_dict'])
    model.eval()
    print('Model load successful~!')

    real_imgs = [str(image_path)]

    for i in range(len(real_imgs)):
        # org_img = cv2.imread(real_imgs[i])
        # cv2.imwrite(
        #   str(Path('input_save/'+real_imgs[i].split('/')[-1])), org_img)
        print('---------------------------------------------------')
        print('  img file : ', real_imgs[i].split('/')[-1])

        pred = rembg.remove_background(
            real_imgs[i], model, alpha_matting=False)

        out_file_path = task_folder_path/Path(
            f"pred_{real_imgs[i].split('/')[-1].split('.')[0]}.png")
        pred.save(out_file_path, 'PNG')
        print('---------------------------------------------------')

    return out_file_path


if __name__ == '__main__':
    run_rembg()
