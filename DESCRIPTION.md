## **Image Keypoint Processing**

- **Description**: 이 엔드포인트는 이미지 처리 및 키포인트 추출 작업을 수행합니다. 클라이언트는 이미지 파일과 함께 작업에 필요한 기타 파라미터들을 업로드합니다.
- **Request Parameters**:
    - **`zip_file`**: image, depth, ply 파일을 포함한 ZIP 파일 (업로드)
    - **`clothes_type`**: 의류 타입 (쿼리 파라미터)
    - **`model_version`**: 모델 버전 (쿼리 파라미터)
- **Response**: 처리된 이미지 파일 (PNG 형식)
    - **Example:**
        
        ![https://i.imgur.com/OnO1yhum.png](https://i.imgur.com/OnO1yhum.png)
        

## **Background Removal**

- **Description**: 이 엔드포인트는 이미지의 배경을 제거하는 작업을 수행합니다. 클라이언트는 배경을 제거할 대상 이미지 파일을 업로드합니다.
- **Request Parameters**:
    - **`image_file`**: 배경을 제거할 이미지 파일 (업로드)
        - **Example:**
        
        ![https://i.imgur.com/Q59kMlmm.jpg](https://i.imgur.com/Q59kMlmm.jpg)
        
- **Response**: 배경이 제거된 이미지 파일 (PNG 형식)
    - **Example:**
    
    ![https://i.imgur.com/CAooSIFm.png](https://i.imgur.com/CAooSIFm.png)
    

## **Inpainting with Stable Diffusion**

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