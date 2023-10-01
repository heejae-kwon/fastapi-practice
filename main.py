from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, Body, status
from fastapi.responses import FileResponse
from typing import Annotated

from rembg.rembg import run_rembg
from inpainting.inpainting import run_in_painting_with_stable_diffusion
from img_kpt.img_kpt import run_img_kpt_processing

import zipfile


api_desc = ""
with open(Path('DESCRIPTION.md'), 'r') as f:
    # api_desc = f.read()
    pass

app = FastAPI(title="Model API",
              description=api_desc,
              version="0.0.1",
              # openapi_tags=tags_metadata
              )


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    pass


@app.get("/api")
async def root():
    return {"message": "Hello API"}


@app.post('/api/v1.0/img-kpt',
          status_code=status.HTTP_201_CREATED,
          response_class=FileResponse,
          responses={
              201: {
                  "content": {"image/png": {"example": "(binary image data)"}},
                  "description": "Created",
              },
              422: {
                  "content": {"application/json": {
                      "example": {"errors": [
                          {
                              "title": "Failed internal process"
                          }
                      ]}
                  }},
                  "description": "Internal process error"},
          })
async def image_keypoint(
    zip_file: Annotated[UploadFile, File(
        example="",
        description="image, depth, ply 파일을 포함한 ZIP 파일")],
    clothes_type:  Annotated[int, Query(description="옷 종류")],
    model_version: Annotated[int, Query(description="모델버전")]

):
    """
- **Description**: 이 엔드포인트는 이미지 처리 및 키포인트 추출 작업을 수행합니다. 클라이언트는 이미지 파일과 함께 작업에 필요한 기타 파라미터들을 업로드합니다.

- **Request Parameters**:
    - **`zip_file`**: image, depth, ply 파일을 포함한 ZIP 파일 (업로드)
    - **`clothes_type`**: 의류 타입 (쿼리 파라미터)
    - **`model_version`**: 모델 버전 (쿼리 파라미터)

- **Response**: 처리된 이미지 파일 (PNG 형식)
    - **Example:**

        ![https://i.imgur.com/OnO1yhum.png](https://i.imgur.com/OnO1yhum.png)
    """
    if not zip_file:
        return {"errors": [
            {
                "title": "No zip file"
            }
        ]}

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path(
        'img_kpt/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name

    task_folder_path.mkdir(parents=True, exist_ok=True)

    zip_file_path = task_folder_path / 'zip_file.zip'
    with zip_file_path.open("wb") as f:
        f.write(await zip_file.read())

    # Unzip files
    with zipfile.ZipFile(zip_file_path, 'r') as unzip_file:
        unzip_file.extractall(task_folder_path)

    run_img_kpt_processing(task_folder_path=task_folder_path,
                           tflite_model_path=Path(
                               'img_kpt/model/test_hrnet.tflite'),
                           clothes_type=clothes_type,
                           model_version=model_version)
    # Return the result file as a response
    result_image_path = task_folder_path / 'result_image_v1.png'
    return FileResponse(
        result_image_path,
        status_code=status.HTTP_201_CREATED,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_img_kpt_image_v1.png"}
    )


@app.post('/api/v1.0/rembg',
          status_code=status.HTTP_201_CREATED,
          response_class=FileResponse,
          responses={
              201: {
                  "content": {"image/png": {"example": "(binary image data)"}},
                  "description": "Created",
              },
              422: {
                  "content": {"application/json": {
                      "example": {"errors": [
                          {
                              "title": "Failed internal process"
                          }
                      ]}
                  }},
                  "description": "Internal process error"},
          })
async def remove_background(
    image_file: Annotated[UploadFile, File(media_type="image/png",
                                           description="배경제거 할 이미지")]
):
    """
- **Description**: 이 엔드포인트는 이미지의 배경을 제거하는 작업을 수행합니다. 클라이언트는 배경을 제거할 대상 이미지 파일을 업로드합니다.

- **Request Parameters**:
    - **`image_file`**: 배경을 제거할 이미지 파일 (업로드)
        - **Example:**

            ![https://i.imgur.com/Q59kMlmm.jpg](https://i.imgur.com/Q59kMlmm.jpg)
- **Response**: 배경이 제거된 이미지 파일 (PNG 형식)
    - **Example:**

        ![https://i.imgur.com/CAooSIFm.png](https://i.imgur.com/CAooSIFm.png)
"""
    if not image_file:
        return {"error": "No file uploaded"}

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path(
        'rembg/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path('img_file.jpg')
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    result_image_path = run_rembg(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        status_code=status.HTTP_201_CREATED,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_rembg_image_v1.png"}
    )


@app.post('/api/v1.0/inpainting',
          status_code=status.HTTP_201_CREATED,
          response_class=FileResponse,
          responses={
              201: {
                  "content": {"image/png": {"example": "(binary image data)"}},
                  "description": "Created",
              },
              422: {
                  "content": {"application/json": {
                      "example": {"errors": [
                          {
                              "title": "Failed internal process"
                          }
                      ]}
                  }},
                  "description": "Internal process error"},
          })
async def inpainting(
    prompt: Annotated[str, Body(description="stable diffusion 프롬프트")],
    image_file: Annotated[UploadFile,  File(media_type="image/png",
                                            description="이미지파일")],
    mask_file: Annotated[UploadFile, File(media_type="image/png",
                                          description="마스크이미지")]
):
    """
- **Description**: 이 엔드포인트는 Stable Diffusion을 사용하여 이미지를 보정하는 작업을 수행합니다. 클라이언트는 텍스트 프롬프트, 이미지, 그리고 마스크 이미지 파일을 업로드합니다.

- **Request Parameters**:
    - **`prompt`**: 텍스트 프롬프트 (바디)
        - **Example: “***a mecha robot sitting on a bench***”**
    - **`image_file`**: 작업에 사용할 이미지 파일 (업로드)
        - **Example:**

            ![https://i.imgur.com/nlYnq9hm.png](https://i.imgur.com/nlYnq9hm.png)

    - **`mask_file`**: 마스크 이미지 파일 (업로드)
        - **Example:**

            ![https://i.imgur.com/twjL8uBm.png](https://i.imgur.com/twjL8uBm.png)

- **Response**: 보정된 이미지 파일 (PNG 형식)
    - **Example:**

        ![https://i.imgur.com/rKYbwRpl.png](https://i.imgur.com/rKYbwRpl.png)
"""
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path(
        'inpainting/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path('diffusion_image.png')
    mask_file_path = task_folder_path/Path('diffusion_mask_image.png')
    result_image_path = task_folder_path/Path('diffusion_result_image.png')

    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    with mask_file_path.open("wb") as f:
        f.write(await mask_file.read())

    run_in_painting_with_stable_diffusion(img_path=image_file_path,
                                          mask_path=mask_file_path,
                                          result_path=result_image_path,
                                          prompt=prompt)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        status_code=status.HTTP_201_CREATED,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_inpainting_image_v1.png"}
    )
