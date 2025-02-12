---
layout: single
title:  "[OCR/AI] 손글씨에 강한 Google Cloud Vision AI 사용법 총정리"
categories:
  - data science
tags:
  - AI
  - OCR
  - Google
---

# 0. 들어가며
안녕하세요! 2025년 첫 포스팅은 OCR 관련 포스팅으로 돌아왔습니다.

[23년도말 OCR 서비스 비교평가 포스팅](https://sooeun67.github.io/data%20science/ocr-comparison/)이 제 예상보다 많은 분들에게 사랑(?) 받아 얼떨떨했는데요
약 1년이 지난 이 시점에, 좀더 어려운 난이도의 표와 손글씨가 혼재되어 있는 이미지의 데이터를 테이블화 시켜야 하는 Task를 수행하며 OCR서비스들을 다시 한 번 사용하게 되었습니다. 그 중, 꽤 괜찮은 성능을 보여준 Google Vision AI 사용법에 대해 차근차근 정리해보았습니다. OCR 관련 [Google의 official documentation ]([https://](https://cloud.google.com/vision/docs/handwriting?hl=ko))에서 확인해보실 수 있습니다. 

*저는 Colab 노트북 환경에서 진행했습니다.*

# Google Cloud Vision API

## **📌 Google Cloud Vision API 소개**

**Google Cloud Vision API**는 Google Cloud에서 제공하는 강력한 **컴퓨터 비전 API**로, 다양한 이미지 분석 기능을 지원하는 인공지능 서비스입니다. 머신러닝 기반의 Vision 모델을 활용하여 **이미지 속 객체, 텍스트, 라벨, 얼굴, 로고, 랜드마크 등**을 자동으로 인식하고 분석할 수 있습니다. 아래와 같이 여러 기능을 제공하고 있고, 저는 `Text_Detection` 과 `Document_Text_Detection` 을 테스트 했습니다. 

### 🚀 주요 기능
✅ OCR (Optical Character Recognition) 기능
- <span style="color:blue">**TEXT_DETECTION**:</span> 이미지 속 텍스트(활자체) 추출
- <span style="color:blue">**DOCUMENT_TEXT_DETECTION**: </span> 문서 내 텍스트 및 레이아웃 분석
- 손글씨 인식 가능 (비정형 데이터도 일부 지원)

✅ 객체 및 이미지 분석
- **LABEL_DETECTION**: 이미지 속 주요 객체 및 개념 태깅
- **OBJECT_LOCALIZATION**: 이미지 내 여러 객체의 위치와 바운딩 박스 추출
- **LOGO_DETECTION**: 브랜드 및 로고 감지

✅ 얼굴 및 감정 분석
- **FACE_DETECTION**: 얼굴 감지 및 표정, 감정 분석

✅ 문서 및 제품 검색
- **WEB_DETECTION**: 이미지 기반 웹 검색
- **PRODUCT_SEARCH**: 제품 이미지 검색 및 매칭


# 1. API를 통한 호출 방식

### ✅ **1. 기존 API Key 방식**

 **Google Cloud Vision API Key**를 사용하여 **REST API** 방식으로 요청을 보낼 수 있습니다.

즉, **HTTP 요청 (`requests.post()`)을 직접 만들어** Vision API 엔드포인트 (`https://vision.googleapis.com/v1/images:annotate` )로 보냅니다.

- **🔹 API Key 방식의 특징**
    - API Key만 있으면 어디서든 호출 가능
    - `requests.post()`를 통해 **수동으로 JSON 요청 생성 및 전송**
    - 응답이 JSON 형태로 반환됨 (`response.json()`)

```python
import base64
import requests
import json

# ✅ API 엔드포인트 및 API 키 설정
API_KEY = google_api_key
VISION_API_URL = google_api_url

# ✅ OCR 처리할 이미지 파일 경로
image_path = "/content/sample_imag.jpeg"

# ✅ Step 1: 이미지 파일을 Base64 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ✅ Step 2: Google Vision API 요청 데이터 생성
def create_request(image_base64):
    return {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [
                    {"type": "DOCUMENT_TEXT_DETECTION"},  # 텍스트 인식 (OCR)
                ],
            }
        ]
    }

# ✅ Step 3: Google Vision API 호출
def call_google_vision_api(image_path):
    # 이미지 Base64 변환
    image_base64 = encode_image(image_path)

    # 요청 데이터 생성
    request_data = create_request(image_base64)

    # API 호출
    response = requests.post(VISION_API_URL, json=request_data)

    # 응답 처리
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ API 요청 실패! 상태 코드: {response.status_code}")
        print(response.text)
        return None

# ✅ Step 4: OCR 결과 저장 및 출력
def save_and_print_ocr_results(ocr_response, output_text_path="full_text.txt", output_json_path="ocr_result.json"):
    if not ocr_response:
        print("❌ OCR 결과가 없습니다.")
        return

    # OCR 결과에서 텍스트 추출
    extracted_text = []
    for annotation in ocr_response["responses"][0].get("textAnnotations", []):
        extracted_text.append(annotation["description"])

    # 전체 텍스트 저장
    full_text = "\n".join(extracted_text)

    # ✅ 결과 출력
    print("\n🔹 Extracted OCR Text:\n")
    print(full_text)

    # ✅ 텍스트 저장
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(full_text)
    print(f"\n✅ OCR 텍스트 저장 완료: {output_text_path}")

    # ✅ JSON 저장 (전체 응답)
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(ocr_response, json_file, indent=4, ensure_ascii=False)
    print(f"✅ OCR 결과 JSON 저장 완료: {output_json_path}")

# ✅ 실행
ocr_response = call_google_vision_api(image_path)
save_and_print_ocr_results(ocr_response)
```

# 2. 서비스 계정 인증 방식을 통한 사용 (좀더 안정적)

API호출 방식 보다 좀더 안정적이라고 보면 됩니다.

### 🔹 **필요한 설정**

✅ Google Cloud 프로젝트 생성 → Vision API 활성화

✅ 서비스 계정(JSON 키) 생성 → Google Drive에 업로드

✅ Colab에서 JSON 키 불러오기 → 환경변수 설정

### 🔹 **OCR 실행**

✅ `google-cloud-vision` 설치 후 실행

✅ 이미지에서 **텍스트 추출 및 시각화 가능**

## 🎯 **이제 실행해봅시다! 🚀**

1️⃣ **Google Drive에 JSON 키 업로드**

2️⃣ **Colab에서 마운트 → 키 복사**

3️⃣ **Google Vision API 실행**

4️⃣ **OCR 결과 확인 및 시각화**

## 2-1. ✅ **Colab에서 Google Cloud Vision API 사용 방법**

Colab 노트북에서 Google Vision API를 사용하려면 **Service Account 인증 방식**을 사용해야 합니다.

---

## 🚀 **1. Google Cloud 프로젝트 설정**

### **1️⃣ Google Cloud 프로젝트 생성 및 API 활성화**

1. [**Google Cloud Console**](https://console.cloud.google.com/) 접속
2. **새 프로젝트 생성** (또는 기존 프로젝트 사용)

![1](/assets/img/2025-02-07-google-vision-ai/1.png)

두 가지 세팅을 해주셔야 하는데, `결제` 에서 카드 등록 해놓아야 하고(결제 부분은 생략하겠음!)

그 다음엔 `IAM 및 관리자` 에 들어가서 서비스 계정을 발급받아야 하는데 아래에서 자세히 설명하겠습니다

![2](/assets/img/2025-02-07-google-vision-ai/2.png)

**아, 그 전에 한 가지 해야할거!**

**콘솔 최상단에 있는 검색창에 "Cloud Vision API" 검색 후 `Cloud Vision API` 를 활성화 시켜주어야합니다.**

![3](/assets/img/2025-02-07-google-vision-ai/3.png)

아래와 같이 `API 사용 설정됨` 으로 보이면 설정 완료!

![4](/assets/img/2025-02-07-google-vision-ai/4.png)

---

## 🚀 **2. Service Account(JSON 키) 생성**

### **2️⃣ 서비스 계정 생성 및 JSON 키 다운로드**

1. Google Cloud IAM 관리 이동 하면 아래와 같은 개요 페이지가 보일 겁니다

![2-1](/assets/img/2025-02-07-google-vision-ai/2-1.png)


1. **왼쪽 메뉴 중 중간쯤 있는 "서비스 계정 만들기"** 클릭
    
    ![2-2](/assets/img/2025-02-07-google-vision-ai/2-2.png)

    
2. **서비스 계정 이름 설정** (예: `vision-api-access`)
    
    ![2-3](/assets/img/2025-02-07-google-vision-ai/2-3.png)

    
3. **역할(Role) 추가**:
    - **"Cloud Vision API User"** 선택 해도 되고, 저는 그냥 개인 계정이라서 “소유자”로 지정했어요
        
        ![2-4](/assets/img/2025-02-07-google-vision-ai/2-4.png)

        
4. **계속 → 완료 → 생성된 서비스 계정 클릭**
5. **방금 생성한 서비스 계정의 맨 우측에 작업을 누르면 `키 관리` 가 보이실 거에요
"키 관리" > "키 추가" > "새 키 만들기"**
    
    ![2-5](/assets/img/2025-02-07-google-vision-ai/2-5.png)
    ![2-6](/assets/img/2025-02-07-google-vision-ai/2-6.png)

    
6. **JSON 형식 선택 후 다운로드**
    - `vision-api-key.json` 파일이 다운로드됨

비공개 키는 안전한 곳에 저장해두세요!

자 이제 환경 세팅은 다 끝났고, Colab 으로 돌아가서 API를 본격적으로 써봅시다

---

## 🚀 **3. Colab 환경에서 Google Cloud Vision API 설정**

### **3️⃣ Google Drive에 서비스 계정 키 업로드**

1. Google Drive에 **`vision-api-key.json`** 업로드

```python

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "**vision-api-key.json"**

```

---

## 🚀 **4. Google Cloud Vision API 클라이언트 설치 및 실행**

### **4️⃣ `google-cloud-vision` 패키지 설치**

```python

!pip install --upgrade google-cloud-vision
```

---

## 🚀 **5. Google Cloud Vision API 사용하기**

### **5️⃣ Python 코드 실행 (OCR 텍스트 추출)**

```python
import io
import os
import json
from google.cloud import vision
from google.protobuf.json_format import MessageToDict  # JSON 변환 모듈

# ✅ 1. 서비스 계정 인증 설정 (Colab 환경)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/vision-api-key.json"

# ✅ 2. Vision API 클라이언트 생성
client = vision.ImageAnnotatorClient()

# ✅ 3. OCR 적용할 이미지 경로
image_path = "/content/sample-image.png"  # ✅ OCR 적용할 이미지

# ✅ 4. 이미지 로드
with io.open(image_path, "rb") as image_file:
    content = image_file.read()
image = vision.Image(content=content)

# ✅ 5. OCR 요청 (문서 내 텍스트 검출)
response = client.document_text_detection(image=image)

# ✅ 6. OCR 결과 추출 및 출력
if response.text_annotations:
    extracted_text = response.text_annotations[0].description  # ✅ 전체 텍스트 가져오기
    print("\n🔹 Extracted OCR Text:\n")
    print(extracted_text)

    # ✅ 7. JSON 변환 후 저장 (🚨 기존 오류 해결)
    response_dict = MessageToDict(response._pb)  # ✅ `_pb` 사용하여 변환

    # ✅ 8. OCR 결과 저장 (텍스트)
    with open("ocr_result.txt", "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)
    print(f"\n✅ OCR 텍스트 저장 완료: ocr_result.txt")

    # ✅ 9. OCR 전체 결과 저장 (JSON)
    with open("ocr_result.json", "w", encoding="utf-8") as json_file:
        json.dump(response_dict, json_file, indent=4, ensure_ascii=False)
    print(f"\n✅ OCR 결과 JSON 저장 완료: ocr_result.json")

else:
    print("\n❌ OCR 결과 없음")

```

---

# 3. 결론/비교

### 📌 **1. 두 코드의 차이점**

| **버전** | **방식** | **인증 방법** | **요청 형식** |
| --- | --- | --- | --- |
| **1번** | `google.cloud.vision` 라이브러리 사용 | ✅ **서비스 계정 JSON 파일** 사용 (`GOOGLE_APPLICATION_CREDENTIALS`) | `vision.ImageAnnotatorClient()` 클라이언트로 직접 호출 |
| **2번** | `requests`를 이용한 REST API 호출 | ❌ **API 키 사용** (`YOUR_GOOGLE_VISION_API_KEY`) | `requests.post()`를 이용한 HTTP 요청 |

### **📌 서비스 계정 인증 방식 vs. API Key 방식 비교**

---

### ✅ **2. 기존 API Key 방식**

기존 코드에서는 **Google Cloud Vision API Key**를 사용하여 **REST API** 방식으로 요청을 보냈습니다.

즉, **HTTP 요청 (`requests.post()`)을 직접 만들어** Vision API 엔드포인트 (`https://vision.googleapis.com/v1/images:annotate` )로 보냈습니다.

- **🔹 API Key 방식의 특징**
    - API Key만 있으면 어디서든 호출 가능
    - `requests.post()`를 통해 **수동으로 JSON 요청 생성 및 전송**
    - 응답이 JSON 형태로 반환됨 (`response.json()`)

### ✅ **3. 현재 서비스 계정 인증 방식**

현재는 **Google Cloud Vision SDK**를 사용하여 **gRPC 또는 HTTP** 방식으로 Google Cloud 서버와 직접 통신합니다.

즉, `vision.ImageAnnotatorClient()`를 통해 API 호출을 간편하게 처리할 수 있습니다.

- **🔹 서비스 계정 인증 방식의 특징**
    - `GOOGLE_APPLICATION_CREDENTIALS` 환경 변수를 통해 인증
    - `client.document_text_detection(image=image)` 한 줄로 API 요청 가능
    - 직접 `requests.post()`로 API를 호출할 필요 없음
    - 응답은 JSON이 아니라 **Python 객체 (`AnnotateImageResponse`)** 형태로 반환됨
    - **Google Cloud SDK 내부적으로 최적화된 방식** (gRPC 사용 가능)

---

### **❓ 4. 기존 `requests.post()` 방식이 필요 없는가?**

✅ **그렇습니다!**

서비스 계정 인증 방식을 사용하면 `requests.post()`와 JSON 요청을 직접 만들 필요가 없습니다.

대신, **Google Cloud SDK의 `vision.ImageAnnotatorClient()`가 모든 요청을 자동으로 처리**합니다.


### 📌 5. 결론: 어떤 방식을 선택해야 할까?
✔ 빠른 테스트 & 소규모 프로젝트 → API 키 방식 사용

✔ 보안이 중요한 환경 → 서비스 계정 인증 방식 사용

✔ 팀원 간 권한 관리 & 대량 데이터 처리 → 서비스 계정 사용 권장

🚀 보안이 중요한 경우, API 키 방식보다는 서비스 계정 인증 방식을 추천합니다 😊


# 마치며

- 훌륭한 성능에 비해 생각보다 Reference 가 많지 않고 depreciated된 내용들이 혼재되어 있어 정리하면 좋을 것 같다는 생각이 들었던 Google Vision AI 서비스를 살펴보았습니다.

# Reference

- [Google Vision AI](https://cloud.google.com/vision/docs/handwriting?hl=ko)