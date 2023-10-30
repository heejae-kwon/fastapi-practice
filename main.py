from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, Body, status
from fastapi.responses import FileResponse
from typing import Annotated

from rembg.rembg import run_rembg
from inpainting.inpainting import run_in_painting_with_stable_diffusion
from img_kpt.img_kpt import run_img_kpt_processing


import zipfile

from strawberry.strawberry import run_strawberry

# Load API description from a markdown file
api_desc = ""
with open(Path('DESCRIPTION.md'), 'r') as f:
    api_desc = f.read()

tags_metadata = [
    {
        "name": "API Reference",
    },
]

app = FastAPI(title="Model API",
              description=api_desc,
              version="1.0.0",
              openapi_tags=tags_metadata
              )


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    pass


@app.get("/",
         include_in_schema=False
         )
async def root():
    return {"message": "Hello API"}


@app.post('/api/v1.0/img-kpt',
          tags=["API Reference"],
          status_code=status.HTTP_200_OK,
          response_class=FileResponse,
          responses={
              200: {
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
        description="ZIP file containing images, depth, and ply files")],
    clothes_type: Annotated[int, Query(description="Clothing type")],
    model_version: Annotated[int, Query(description="Model version")]
):
    """
    - **Description**: This endpoint performs image processing and keypoint extraction. Clients upload image files and other necessary parameters for the operation.

    - **Request Parameters**:
        - **`zip_file`**: ZIP file containing images, depth, and ply files (upload)
        - **`clothes_type`**: Clothing type (query parameter)
        - **`model_version`**: Model version (query parameter)

    - **Response**: Processed image file (PNG format)
        - **Example**:

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
                           clothes_type=clothes_type,
                           model_version=model_version)

    # Return the result file as a response
    result_image_path = task_folder_path / 'result_image_v1.png'
    return FileResponse(
        result_image_path,
        status_code=status.HTTP_200_OK,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_img_kpt_image_v1.png"}
    )


@app.post('/api/v1.0/rembg',
          tags=["API Reference"],
          status_code=status.HTTP_200_OK,
          response_class=FileResponse,
          responses={
              200: {
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
                                           description="Image to remove background from")]
):
    """
    - **Description**: This endpoint removes the background of an image. Clients upload the target image file for background removal.

    - **Request Parameters**:
        - **`image_file`**: Image file to remove the background from (upload)
            - **Example**:

                ![https://i.imgur.com/Q59kMlmm.jpg](https://i.imgur.com/Q59kMlmm.jpg)
    - **Response**: Image file with the background removed (PNG format)
        - **Example**:

            ![https://i.imgur.com/CAooSIFm.png](https://i.imgur.com/CAooSIFm.png)
    """
    if not image_file:
        return {"error": "No file uploaded"}

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path('rembg/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path('img_file.jpg')
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    result_image_path = run_rembg(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        status_code=status.HTTP_200_OK,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_rembg_image_v1.png"}
    )


@app.post('/api/v1.0/inpainting',
          tags=["API Reference"],
          status_code=status.HTTP_200_OK,
          response_class=FileResponse,
          responses={
              200: {
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
    prompt: Annotated[str, Body(description="Stable diffusion prompt")],
    image_file: Annotated[UploadFile, File(media_type="image/png",
                                           description="Image file")],
    mask_file: Annotated[UploadFile, File(media_type="image/png",
                                          description="Mask image")]
):
    """
    - **Description**: This endpoint performs image correction using Stable Diffusion. Clients upload a text prompt, an image, and a mask image file.

    - **Request Parameters**:
        - **`prompt`**: Text prompt (body)
            - **Example**: “***a mecha robot sitting on a bench***”
        - **`image_file`**: Image file for the operation (upload)
            - **Example**:

                ![https://i.imgur.com/nlYnq9hm.png](https://i.imgur.com/nlYnq9hm.png)

        - **`mask_file`**: Mask image file (upload)
            - **Example**:

                ![https://i.imgur.com/twjL8uBm.png](https://i.imgur.com/twjL8uBm.png)
    - **Response**: Corrected image file (PNG format)
        - **Example**:

            ![https://i.imgur.com/rKYbwRpl.png](https://i.imgur.com/rKYbwRpl.png)
    """
    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path(
        'inpainting/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / Path('diffusion_image.png')
    mask_file_path = task_folder_path / Path('diffusion_mask_image.png')
    result_image_path = task_folder_path / Path('diffusion_result_image.png')

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
        status_code=status.HTTP_200_OK,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_inpainting_image_v1.png"}
    )


@app.post('/api/v1.0/strawberry',
          tags=["API Reference"],
          status_code=status.HTTP_200_OK,
          response_class=FileResponse,
          responses={
              200: {
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
async def strawberry(
    image_file: Annotated[UploadFile, File(media_type="image/png",
                                           description="Strawberry image for size measurement")],
    ply_file: Annotated[UploadFile, File(media_type="text/plain",
                                         description="Strawberry ply for size measurement")]
):
    """
    - **Description**: This endpoint measures the size of a strawberry from the provided image.

    - **Request Parameters**:
        - **`image_file`**: Strawberry image for size measurement (upload)
            - **Example**:

                ![https://i.imgur.com/gAQt0lLm.jpg](https://i.imgur.com/gAQt0lLm.jpg)
    - **Response**: Image file with the strawberry's size measurement (PNG format)
        - **Example**:

            ![https://i.imgur.com/DzQazmGm.jpg](https://i.imgur.com/DzQazmGm.jpg)
    """
    if not image_file:
        return {"error": "No file uploaded"}

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = Path(
        'strawberry/temp_process_task_files') / sub_folder_name
    task_id = sub_folder_name
    task_folder_path.mkdir(parents=True, exist_ok=True)

    image_file_path = task_folder_path / 'image' / Path('image.jpg')
    image_file_path.parent.mkdir(parents=True, exist_ok=True)
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    ply_file_path = task_folder_path / 'ply' / Path('ply.ply')
    ply_file_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_file_path.open("wb") as f:
        f.write(await ply_file.read())

    result_image_path = run_strawberry(task_folder_path)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        status_code=status.HTTP_200_OK,
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_strawberry_image_v1.png"}
    )
