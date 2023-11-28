---
layout: single
title:  "[OCR/AI] 네이버 CLOVA OCR 영수증 모델 API 신청 사용 가이드"
categories:
  - data science
tags:
  - AI
  - OCR
---

Naver Clova OCR에는 기본 OCR 모델부터 영수증, 사업자등록증, 명함 등 특정 서류에 특화된 Document OCR 모델도 제공하며, 사용하는 특화 모델에 따라 제공하는 Feature가 다릅니다. 
이 중 영수증 모델을 사용하기 위한 API 신청 절차를 순서대로 진행한 포스팅입니다. 특화 모델별 상세한 내용은 CLOVA OCR Document API를 참고해 주십시오.

## 유의사항
------

1. **<span style="color:red">도메인 생성 후 “중지” 기능이 없으므로, 사용 후 꼭 “삭제” 하세요 — 도메인 유지 비용 발생합니다</span>**
2. **<span style="color:red">“중지” 후 다시 도메인 생성 시 새로운 Secret Key & API Invoke URL 필요</span>**

## 1. 네이버 클라우드에 로그인 후 [CLOVA OCR 서비스](https://console.ncloud.com/ocr/domain) 클릭 → “특화 모델 설정” 클릭

![1](/assets/img/2023-07-25-naver-clova-receipt-ocr/1.png)
## 2. 좌측 상단의 “특화 모델 신청” 클릭 후 아래와 같은 팝업 창에 내용 기입

- 특화 모델: 영수증
- 사용 목적, 사용 시기, 예상 사용량(수집된 영수증 데이터 양 기반으로), 상세 내용 기입 후 “확인”

![2](/assets/img/2023-07-25-naver-clova-receipt-ocr/2.png)

신청이 완료 되면, “**신청 상태**” 가 <span style="color:green">“**승인**”</span>으로 보일 겁니다.

## 3. “도메인 생성” 클릭 → “특화 모델 도메인 생성” 클릭

![3](/assets/img/2023-07-25-naver-clova-receipt-ocr/3.png)

아래 정보를 바탕으로 도메인을 생성 합니다.
![4](/assets/img/2023-07-25-naver-clova-receipt-ocr/4.png)

## 4. 생성한 도메인을 API 와 연동하기

- Secret Key 와 API Invoke URL 두 정보가 필요합니다. API Getaway 는 자동 연동으로 둘 것이므로 따로 수정 하거나 수동 연동할 필요 없음.

![5](/assets/img/2023-07-25-naver-clova-receipt-ocr/5.png)

Secret Key **“생성”** 클릭

![6](/assets/img/2023-07-25-naver-clova-receipt-ocr/6.png)
![7](/assets/img/2023-07-25-naver-clova-receipt-ocr/7.png)


해당 Key 와 URL을 사용해서 API 호출에 사용 하면 됩니다.

## API 호출 건당 OCR 이용 예상 요금
---

- **영수증 1,000건 있다면 ⇒ Standard Plan (180,000원) 으로 첫 달 이용하고, 그 이후부턴 Basic (18,000원)이용하면 될듯**
- 도메인 기본 유지 비용이 일할 계산되어 발생 하므로, 단기간에 사용하고 도메인을 삭제하는 것이 경제적
    - 예) Basic Plan 사용시: 한 달간 10일만 사용하고 300건 이하로 이용한다면 ⇒ 18,000원/30일 * 10일 = 약 6,000원 예상 (*부가세별도*)
    - 예) Standard Plan 사용시: 한 달 내내 도메인 띄워놓고 3,002건 사용한다면 ⇒ 180,000원 + 80*2건 = 약 180,160원 예상  (*부가세별도*)

| Plan 타입 | 기본 유지 비용(월) | 과금 구간 | 추가 과금 (1개 호출수당) |
| :---: | :---: | :---: | :---: |
| Basic | 18,000원 | 300건 이하 | 100원 |
| Standard | 180,000원 | 3,000건 이하 | 80원 |
| Adavanced | 580,000원 | 15,000건 이하 | 50원 |

**부가세 별도*

- [요금표](https://www.ncloud.com/product/aiService/ocr)
- [요금 계산기](https://www.ncloud.com/charge/calc/ko)