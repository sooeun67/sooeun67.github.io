---
layout: single
title:  "[OCR/AI] Azure 문서 인텔리전스 API 신청 가이드 및 사용하기(feat. 코랩)"
categories:
  - data science
tags:
  - AI
  - OCR
toc: true
toc_sticky: true
---

안녕하세요, 이번 포스팅에서는 최근 ChatGPT 와의 협업 등 언어 AI 서비스에서 큰 두각을 보이고 있는 **Azure** 의 OCR API를 사용하여 테스트 이미지를 인식해보겠습니다. 
OCR 테스트를 위해서는 먼저 Azure 에 회원 가입 후 Azure Portal 에 접속하여 리소스 및 구독 설정 완료, 키 생성과 같이 API 사용을 위한 준비를 마치고, 
구글 Colab(jupyter notebook) 환경에서 테스트를 진행해 보고자 합니다. 
8가지 이상의 여러 OCR 서비스들을 테스트 해보았을 때, Azure Document AI 서비스는 11월 말까지도 [`v4.0`](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/changelog-release-history?view=doc-intel-4.0.0&tabs=csharp)을 출시하는 것처럼 지속적으로 release를 하고 있는 것으로 보이며,
다양한 언어 모델을 지원하고 있어 영어와 한국어 모두 준수한 성능을 보이고 있다고 생각됩니다. Azure 서비스를 기존에 사용해보셨던 분들이시라면, Document Intelligence 서비스를 굉장히 쉽게 사용하실 거라 생각되고,
만약 Azure 가 처음이시라면 [Document Intelligence](https://azure.microsoft.com/ko-kr/products/ai-services/ai-document-intelligence) 자체는 사용하기 어렵지 않지만, Azure 서비스를 사용하는 데 필수적인 리소스, 권한 할당, 구독 등 전반적인 계정 관리가 낯설어 이해하는 데 조금 시간이 걸릴 수 있을 것 같습니다(저 또한 그랬고요).

> 이전에는 Form Recognizer 라고 불렸었는데, 이제는 Document Intelligence로 부르는 것 같아요

![logo](/assets/img/2023-12-11-ocr-azure-document-intelligence/azure_logo.png)

# Azure의 OCR 서비스

----

Azure 에서 제공하는 OCR 서비스는 크게 두가지로 나눌 수 있습니다.
1. Computer Vision
   1. 단순 추출에 가까움
2. Document Intelligence: 1번 보다 고도화된 서비스로, MS에서 더 주력하고 있는, 단순 추출을 넘어서 entity 제공, language detection 등 복합적인 OCR 서비스를 제공합니다.


Azure AI 문서 인텔리전스는 지능형 문서 처리 솔루션을 빌드할 수 있는 클라우드 기반 Azure AI 서비스입니다. 
다양한 데이터 형식에 걸친 방대한 양의 데이터가 양식과 문서에 저장됩니다. 문서 인텔리전스를 사용하면 데이터가 수집 및 처리되는 속도를 효과적으로 관리할 수 있으며 향상된 운영, 정보에 입각한 데이터 기반 의사 결정 및 인식 가능한 혁신의 핵심입니다.

Azure AI 문서 인텔리전스에는 기본적인 문서 추출 모델뿐만 아니라 다양한 pre-built model(특화모델)들이 제공됩니다.

![models](/assets/img/2023-12-11-ocr-azure-document-intelligence/models.png)



# 

----


5건의 이미지와 서비스별, 이미지별 소요시간 비교 그래프는 OCR 비교평가 포스팅에 별도로 정리해놓았으니, 
서비스별 비교평가가 필요한 독자 분들은 [해당 포스팅](https://sooeun67.github.io/data%20science/ocr-comparison/)에서 확인하시면 됩니다. 

