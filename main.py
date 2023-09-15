from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Depends, Query
from fastapi.responses import FileResponse
from typing import Annotated

from fastapi_test.rgb_to_json import run_temp_processing
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
        "description": "The first entry point",
    },
    {
        "name": "temp_processing",
        "description": "Run the temp process",
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


@app.get("/", tags=["root"])
async def root():
    return {"message": "Hello API"}


TEMP_UPLOAD_FOLDER = Path('./temp_process_task_files')


@app.post('/api/server/temp_processing', tags=["temp_processing"])
async def temp_processing(
    zip_file: Annotated[UploadFile, File(description="Zip file that contains image,depth,ply files")],
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

    await run_temp_processing(task_folder_path=task_folder_path,
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
