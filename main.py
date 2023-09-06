from datetime import datetime
from pathlib import Path

from fastapi_test.image_processing import pre, post, landmark_task_1

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

import json
import zipfile


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello API"}


# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


PRE_UPLOAD_FOLDER = Path('./pre_process_task_files')
POST_UPLOAD_FOLDER = Path('./post_process_task_files')


@app.post('/api/server/pre_processing')
async def pre_processing(image_file: UploadFile):
    if not image_file:
        return JSONResponse(content={'error': 'No file uploaded'}, status_code=400)

    now_date = datetime.now()
    sub_folder_name = now_date.strftime("%Y%m%d%H%M%S%f")
    sub_folder_path = PRE_UPLOAD_FOLDER / sub_folder_name

    sub_folder_path.mkdir(parents=True, exist_ok=True)

    file_path = sub_folder_path / str(image_file.filename)

    with file_path.open('wb') as file:
        file.write(await image_file.read())

    # Assuming you have a pre-processing function
    pre_result = await pre(image_file_path=str(file_path), task_id=sub_folder_name)
    pre_result_json = json.dumps(pre_result)

    return JSONResponse(content=pre_result_json)


@app.post('/api/server/post_processing')
async def post_processing(
    task_id: str = Form(...),
    json_file: UploadFile = File(...),
    zip_file: UploadFile = File(...)
):
    task_folder_path = POST_UPLOAD_FOLDER / task_id
    task_folder_path.mkdir(parents=True, exist_ok=True)

    # Save files
    zip_file_path = task_folder_path / 'zip_file.zip'
    json_file_path = task_folder_path / 'output.json'

    with zip_file_path.open('wb') as zf:
        zf.write(await zip_file.read())

    with json_file_path.open('wb') as jf:
        jf.write(await json_file.read())

    # Unzip files
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        zf.extractall(task_folder_path)

    with json_file_path.open('r') as jf:
        jsonData = json.load(jf)
        await post(task_folder_path, outputData=jsonData['output'])
    await landmark_task_1(task_folder_path)

    # Call post() and landmark_task_1() functions here with task_folder_path

    zip_file_name = f'{task_id}_result.zip'
    zip_file_output_path = task_folder_path / zip_file_name

    with zipfile.ZipFile(zip_file_output_path, 'w') as zf:
        zf.write(zip_file_path, 'zip_file.zip')

    return FileResponse(
        zip_file_output_path,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename={zip_file_name}'}
    )
