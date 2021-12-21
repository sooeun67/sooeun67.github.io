---
layout: single
title:  "[python] googletrans 구글 번역 무료 API로 데이터프레임 한영 번역/변환을 한번에 (번역 시리즈1)"
categories:
  - data science
tags:
  - pandas
---

**구글 번역 API [googletrans](https://github.com/ssut/py-googletrans) 를 통해 데이터 프레임(dataframe)을 한글 <-> 영어/외래어 변환하는 모듈을 개발 하였습니다.**

- 한 문장 한 문장을 실시간 번역하는 예시는 많이 나와있습니다. (이건 다른 포스트에서 다뤄보겠습니다) 
- 전체 데이터 프레임에 있는 데이터들을 한 번에 번역하고 싶어 개발하게 되었습니다. 


## **0. 개발 배경**

- 건설 회사 견적 품명 데이터를 의뢰 받았는데, 한글과 영어가 **혼재**되어 있어 통일 필요
- 샘플 데이터를 받았던 상황이었으므로 분류 모델에 필요한 학습 데이터 부족 -> 따라서 **data augmentation** (데이터 증강) 의 방법으로 사용


## **1. 패키지 설치**

```python
!pip install googletrans==4.0.0-rc1             # package version update
import googletrans
from googletrans import Translator
import pandas as pd
```


## **2. translator 함수** 

```python
def translate_df(df, col, lang='en'):
  """ 데이터프레임(df)을 input으로 받아서 google translator로 번역할 칼럼(col)을 string 타입으로 지정해주면
      한영 번역된 문자열 값이 데이터프레임에 append 됩니다.
      df = input 데이터프레임
      col = 번역하고자 하는 칼럼명 (string type)
      lang = default는 영어(en)로 설정되어 있으며 언어 설정 변경 가능 (string type)  e.g. 일어:ja, 중국어:zh-cn
      지원 언어 리스트: {'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic', 'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian', 'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan', 'ceb': 'cebuano', 'ny': 'chichewa', 'zh-cn': 'chinese (simplified)', 'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian', 'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english', 'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish', 'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian', 'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole', 'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'he': 'hebrew', 'hi': 'hindi', 'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer', 'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao', 'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish', 'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam', 'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian', 'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'or': 'odia', 'ps': 'pashto', 'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi', 'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic', 'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi', 'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish', 'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 'tr': 'turkish', 'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'uz': 'uzbek', 'vi': 'vietnamese', 'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu'}
  """
 
  translator = Translator()
  df['eng'] = [translator.translate(i, src='ko', dest=lang).text for i in df[col]]   # 한영 변환을 새로운 칼럼 'eng'에 담습니다
   
  tmp1 = df.drop(['eng'], axis=1)                                                    # 2개의 temporary 데이터프레임을 생성해 원본과 한영번역 데이터프레임을 합칩니다
  tmp2 = df.drop([col], axis=1)
  tmp2.rename(columns={'eng':'data'}, inplace=True)
  result = pd.concat([tmp1,tmp2], ignore_index=True)
  result = result.drop_duplicates()
  return result                                                                      # 기존 데이터 프레임에 한영 변환된 데이터프레임이 append 되어 리턴
```

## **테스트 결과**

- 좌측 하단 10개 행의 샘플 데이터프레임으로 한영 변환 모듈 테스트 결과가 우측에 나타납니다.
- 테스트하고자 하는 데이터프레임명(`s`) 과 칼럼명(`data`)을 입력 값으로 사용합니다.
  (lang 언어 설정 argument가 따로 지정되지 않는 경우 영어가 default 값)
- **기존 데이터프레임 하단에 한영 변환된 data 값들이 append 되어 새로운 20개 행을 가진 데이터프레임을 리턴합니다**.
- 수행 시간 테스트: 100 개 : 10초 / 400개: 29초

![img](/assets/img/2021-06-10-googletrans-1/test-result.png)