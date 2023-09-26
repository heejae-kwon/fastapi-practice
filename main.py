from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, Body
from fastapi.responses import FileResponse
from typing import Annotated

from rembg.rembg import run_rembg
from inpainting.inpainting import run_in_painting_with_stable_diffusion
from img_kpt.img_kpt import run_img_kpt_processing

import zipfile


description = ""
with open(Path('DESCRIPTION.md'), 'r') as f:
    description = f.read()

tags_metadata = [
    {
        "name": "root",
        "description": "The first entry point of api",
    },
    {
        "name": "img-kpt",
        "description": "Get the size of clothes",
    },
    {
        "name": "rembg",
        "description": "Remove the background of image",
    },
    {
        "name": "inpainting",
        "description": "Change image to another image",
    },
]

app = FastAPI(title="Model API",
              description=description,
              version="0.0.1",
              openapi_tags=tags_metadata
              )


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    pass


@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello API"}


TEMP_UPLOAD_FOLDER = Path('./temp_process_task_files')


@app.post('/api/img-kpt', tags=["img-kpt"],
          response_class=FileResponse,
          responses={
    200: {
        "content": {
            "image/png": {},
        },
        "description": """![bear](https://placebear.com/cache/395-205.jpg)\n
Polar bear image.
        """,
    }
})
async def image_keypoint(
    zip_file: Annotated[UploadFile, File(
        description="Zip file that contains image,depth,ply files")],
    clothes_type:  Annotated[int, Query(description="type of clothes")],
    model_version: Annotated[int, Query(description="version of model")]

):
    if not zip_file:
        return {"error": "No file uploaded"}

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    task_folder_path = TEMP_UPLOAD_FOLDER / sub_folder_name
    task_id = sub_folder_name

    task_folder_path.mkdir(parents=True, exist_ok=True)

    zip_file_path = task_folder_path / 'zip_file.zip'
    with zip_file_path.open("wb") as f:
        f.write(await zip_file.read())

    # Unzip files
    with zipfile.ZipFile(zip_file_path, 'r') as unzip_file:
        unzip_file.extractall(task_folder_path)

    run_img_kpt_processing(task_folder_path=task_folder_path,
                           tflite_model_path=Path('test_hrnet.tflite'),
                           clothes_type=clothes_type,
                           model_version=model_version)
    # Return the result file as a response
    result_image_path = task_folder_path / 'result_image_v1.png'
    return FileResponse(
        result_image_path,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename={task_id}_result_image_v1.png"}
    )


@app.post('/api/rembg', tags=["rembg"],
          response_class=FileResponse,
          responses={
    200: {
        "content": {"image/png": {}},
        "description": "Return the background removed image.",
    }
})
async def remove_background(
    image_file: UploadFile = File(..., media_type="image/png",
                                  description="Image file to remove background")
):
    image_file_path = Path('prof1.jpg')
    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    result_image_path = run_rembg(image_file_path)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename=rembg_image_v1.png"}
    )


@app.post('/api/inpainting', tags=["inpainting"],
          response_class=FileResponse,
          responses={
    200: {
        "content": {"image/png": {}},
        "description": "Return the background removed image.",
    }
})
async def inpainting(
    prompt: Annotated[str, Body(description="Several words")],
    image_file: UploadFile = File(..., media_type="image/png",
                                  description="Image file needs to run stable diffusion"),
    mask_file: UploadFile = File(..., media_type="image/png",
                                 description="Mask image file needs to run stable diffusion")
):
    image_file_path = Path('diffusion_image.png')
    mask_file_path = Path('diffusion_mask_image.png')

    with image_file_path.open("wb") as f:
        f.write(await image_file.read())

    with mask_file_path.open("wb") as f:
        f.write(await mask_file.read())

    result_image_path = await run_in_painting_with_stable_diffusion(image_file_path,
                                                                    mask_file_path,
                                                                    prompt)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename=stable_diffusion_image_v1.png"}
    )
