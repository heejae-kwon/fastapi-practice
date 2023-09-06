from datetime import datetime
from fastapi import FastAPI, UploadFile
from pathlib import Path
from fastapi_test.image_processing import pre

import json

PRE_UPLOAD_FOLDER = './pre_process_task_files'
POST_UPLOAD_FOLDER = './post_process_task_files'
# app.config['PRE_UPLOAD_FOLDER'] = PRE_UPLOAD_FOLDER
# app.config['POST_UPLOAD_FOLDER'] = POST_UPLOAD_FOLDER


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello API"}


# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
@app.post('/api/server/pre_processing')
async def pre_processing(image_file: UploadFile):
    # check if the post request has the file part
    '''
    if 'image_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    '''
    # if user does not select file, browser also
    # submit a empty part without filename
    '''
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    # if file and allowed_file(file.filename):
    '''
    if image_file.filename == '':
        return

    file = await image_file.read()
    if file:
        filename = (image_file.filename)
        nowDate = datetime.now()
        sub_folder_name = nowDate.strftime("%Y%m%d%H%M%S%f")
        sub_folder_path = Path('PRE_UPLOAD_FOLDER' +
                               '/' + sub_folder_name)
        '''
        if not os.path.exists(sub_folder_path):
            try:
                os.makedirs(sub_folder_path)
            except Exception as e:
                print(e)
        '''
        sub_folder_path.mkdir(exist_ok=True)

        filename_path = sub_folder_path / Path(filename)

        # file.save(os.path.join(sub_folder_path, filename))
        with filename_path.open("wb") as fp:
            fp.write(file)
        print('check point')
        pre_result = pre(image_file_path=str(filename_path),
                         # (sub_folder_path + "/" + filename)
                         task_id=sub_folder_name)
        pre_result_json = json.dumps(pre_result)
        return pre_result_json
