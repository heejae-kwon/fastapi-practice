from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, Body
from fastapi.responses import FileResponse
from typing import Annotated

from pydantic import Field
from rembg.rembg import run_rembg
from in_painting_with_stable_diffusion_using_diffusers.in_painting_with_stable_diffusion_using_diffusers import run_in_painting_with_stable_diffusion
from rgb_to_json.rgb_to_json import run_temp_processing
import zipfile


description = """
ChimichangApp API helps you do awesome stuff. 🚀

## Items

You can **read items**.

## Users

You will be able to:

* **Create users** (_not implemented_).
* **Read users** (_not implemented_).
"""

tags_metadata = [
    {
        "name": "root",
        "description": "The first entry point of api",
    },
    {
        "name": "temp_processing",
        "description": "Get the size of clothes",
    },
    {
        "name": "rembg",
        "description": "Remove the background of image",
    },
    {
        "name": "stable-diffusion",
        "description": "Change image to another image",
    },
]

app = FastAPI(title="ChimichangApp",
              description=description,
              summary="Deadpool's favorite app. Nuff said.",
              version="0.0.1",
              terms_of_service="http://example.com/terms/",
              contact={
                  "name": "Deadpoolio the Amazing",
                  "url": "http://x-force.example.com/contact/",
                  "email": "dp@x-force.example.com",
              },
              license_info={
                  "name": "Apache 2.0",
                  "identifier": "MIT",
              },
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


@app.post('/api/server/temp_processing', tags=["temp_processing"],
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
async def temp_processing(
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

    run_temp_processing(task_folder_path=task_folder_path,
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


@app.post('/api/server/rembg', tags=["rembg"],
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


@app.post('/api/server/stable-diffusion', tags=["stable-diffusion"],
          response_class=FileResponse,
          responses={
    200: {
        "content": {"image/png": {}},
        "description": "Return the background removed image.",
    }
})
async def in_painting_with_stable_diffusion(
    prompt : Annotated[str, Body(description="Several words")],
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

    result_image_path = await run_in_painting_with_stable_diffusion(
        image_file_path, mask_file_path, prompt)

    return FileResponse(
        result_image_path,
        media_type='image/png',
        headers={
            "Content-Disposition": f"attachment; filename=stable_diffusion_image_v1.png"}
    )
