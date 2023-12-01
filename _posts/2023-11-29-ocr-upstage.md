---
layout: single
title:  "[OCR/Python] Upstage OCR 모델 API 신청부터 직접 사용해보자(코랩)"
categories:
  - data science
tags:
  - AI
  - OCR

---

안녕하세요, 이번 포스팅에서는 OCR 의 "신흥강자" **Upstage** 의 OCR API를 사용하여 테스트 이미지를 인식해보겠습니다. OCR 테스를 위해서는 먼저 Upstage 에 회원 가입 후 Console에 접속하여 API 사용을 위한 준비를 마치고, Colab과 같은 jupyter notebook 환경에서 테스트를 진행해 보고자 합니다. 8가지 이상의 여러 OCR 서비스들을 테스트 해보았을 때, Upstage의 Document AI 서비스는 신생 서비스이지만, 신청 및 사용 절차가 상대적으로 간편하여 사용 복잡도나 난이도가 쉬운 편이라고 생각됩니다.

# 0. Upstage OCR API 신청 하기

--------

## 1. 회원가입

[Upstage 홈페이지](https://www.upstage.ai/)에 접속하면, Document AI 라는 서비스가 보입니다. 

![logo](/assets/img/2023-11-29-ocr-upstage/upstage.png)

해당 서비스를 클릭하면, 아래와 같은 **"무료 크레딧 증정 이벤트"** 팝업창을 볼 수 있을 겁니다. 

![credit](/assets/img/2023-11-29-ocr-upstage/credit.png)

## 2. 콘솔 이동

팝업창을 클릭하면 회원가입을 진행한 후, [로그인](https://console.upstage.ai/login/terms)을 하면, Console로 이동하게 됩니다.

![console](/assets/img/2023-11-29-ocr-upstage/console.png)

## 3. 결제 수단 등록

서비스 사용을 위해서는 카드 정보를 입력하여 결제 수단을 먼저 등록해야 합니다. `Billing` 페이지에서는 각 프로젝트 별로 서비스 이용에 대한 요금이 나옵니다. 

![credit](/assets/img/2023-11-29-ocr-upstage/credit.png)

## 4. 프로젝트 생성

Upstage의 OCR 서비스인 **Document AI** 를 사용하기 위해 **Create New Project**를 클릭해 새로운 프로젝트를 생성합니다. 

아래 두 프로젝트는 기존에 제가 생성해놓은 프로젝트입니다. 새롭게 생성된 프로젝트는 아래 리스트에 추가되겠죠?

<img src="/assets/img/2023-11-29-ocr-upstage/create_project.png" width="300" height="150"/>

Project명은 랜덤하게 assign 되나 원하는 이름으로 변경해도 됩니다. Create 를 클릭해 생성합니다.

![create_2](/assets/img/2023-11-29-ocr-upstage/create_2.png)

## 5. Token 생성

새로운 프로젝트가 생성되면, 해당 프로젝트 페이지로 자동으로 이동하게 됩니다. **Access Token**을 클릭해 새로운 토큰을 생성합니다.

![access_token](/assets/img/2023-11-29-ocr-upstage/access_token.png)

토큰명 또한 자동으로 생성됩니다. **Create** 을 눌러 생성합니다.

<p align="center">
  <img src="/assets/img/2023-11-29-ocr-upstage/generate_token.png" width="150" height="150" />
</p>

이제 토큰 Key 정보가 생성된 것을 확인할 수 있습니다. 이 토큰을 가지고, 코랩이나 주피터 노트북에서 API를 호출 하면 됩니다.

> Token 키는 공개되지 않도록 유의 해야 합니다~

<p align="center">
  <img src="/assets/img/2023-11-29-ocr-upstage/generate_token_done.png" width="150" height="150" />
</p>

자 이제 API 사용을 위한 모든 준비가 끝났습니다!!! 이제 코랩이나 jupyter notebook 환경으로 가볼게요~

# 1. OCR API 사용 하기

-------

### 0. Basic How-To-Use: From Official Documentation

구글 코랩 환경에서 Python 으로 Upstage의 Document AI API를 호출해보겠습니다. Upstage 에서 공식적으로 안내하는 API 호출 방법은 아래와 같습니다. `api_key` 에는 방금 만든 토큰을, `filename`에는 테스트 하고자 하는 이미지 경로를 넣으면 됩니다. 

```python
import requests

api_key = "YOUR_API_KEY"
filename = "YOUR_FILE_NAME"

url = "https://ap-northeast-2.apistage.ai/v1/document-ai/ocr"
headers = {"Authorization": f"Bearer {api_key}"}
files = {"image": open(filename, "rb")}
response = requests.post(url, headers=headers, files=files)
print(response.json())
```



## 아래는 이미지 1건에 대한 Upstage OCR 인식 결과, 5건의 multiple 이미지에 대한 결과, 그리고 이미지별 소요시간을 출력한 상세 코드 입니다. 

5건의 이미지와 서비스별, 이미지별 소요시간 비교 그래프는 OCR 비교평가 포스팅에 별도로 정리해놓았으니, 서비스별 비교평가가 필요한 독자 분들은 [해당 포스팅](https://sooeun67.github.io/data%20science/ocr-comparison/)에서 확인할 수 있습니다. 







# 2. 유의 사항

------

- Upstage 의 billing system 은 서비스를 "사용한 만큼" 지불합니다. 따라서, project 를 생성하고 매번 삭제할 필요는 없기 때문에, 동일한 프로젝트명과 Token을 사용할 수 있어 편리할 것 같네요.

- 사용량(Usage)은 Console 내 **Usage** 메뉴에서 조회가 가능하며, 보통 01:00 UTC 기준으로 업데이트 된다고 합니다. 그러나, 실시간 이용량 조회가 지원되지 않기 때문에, 단시간 내에 여러 건의 테스트를 진행한다면, 대략적으로 본인이 몇 건을 호출하는지 감을 잡고 진행하는 것이 좋을 것 같다는 생각이 드네요. 

- 더 자세한 정보나 문의는 Upstage에서 운영하는 [Documentation](https://upstage.gitbook.io/console/) 를 참고해주세요.