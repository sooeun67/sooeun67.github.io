---
layout: single
title:  "[python] googletrans 구글 번역 API Ban 밴 당할 때 우회해서 데이터프레임 한영 변환 해결 (번역 시리즈2)"
categories:
  - data science
tags:
  - pandas
  - python
---

[이전 포스트](https://sooeun67.github.io/data%20science/googletrans-1/)는 googletrans 라이브러리를 이용하여 데이터프레임 안에 있는 데이터를 한 번에 한영/영한 변환 하는 방법을 다뤄보았습니다.

**번역기 사용 중 API Ban 이슈 발생하여 한영 dictionary 를 cache file 에 저장하여 사용하는 우회 방식을 선택해 진행해보았습니다.** 


## **0. Issue / Things to Consider**

- 2021.3 API ban 하는 이슈 발생 → 우선 exception 발생하는 경우 API reinitialize 해서 우회 하는 방식 으로 진행 (time.sleep 안 통함)
- Colab 환경에서 테스트 하였기 때문에, 런타임 초기화로 API reinitialize 가능
- stability 위해서는 [Google's official translation API](https://cloud.google.com/translate/docs) 사용 해야함 (유료) -> 따라서 완전한 서비스를 위해 본 라이브러리를 사용한다면 이 우회 방식은 비추천
- Colab 사용 시 런타임 초기화 하면, reinitialize 되기 때문에 ban 되지 않는 운 좋은 케이스라고 생각됨 ( local 에서 호출 반복 하면, ban 될 것으로 보임 )번역에 대한 mapping 정보(한글↔ 영문 변환) 를 cache excel 에 저장하고, exception raised 된 부분부터 다시 translator 함수 호출

 

## **1. 패키지 설치**

```python
import sys
if not 'googletrans' in sys.modules.keys():						
  !pip install googletrans==4.0.0-rc1                           # updated package version
import googletrans
from googletrans import Translator
import pandas as pd
import time
import os
```

## **2. 한글 영문 mapping 정보를 dictionary 에 담기**

```python
cache_path = os.path.join(BASE_PATH, 'cache.xlsx')
if not os.path.exists(cache_path):
  pd.DataFrame({'from': [], 'to': []}).to_excel(cache_path)
 
cache_excel = pd.read_excel(cache_path)
cache_excel

cache = {}
for kor, eng in zip(cache_excel['from'], cache_excel['to']):
  cache[kor] = eng
```

 

## **3. Mapping 여부에 따라 Translator 함수 호출**

한글<->영문 변환 mapping 이 cache 에 되어 있으면 그 정보를 불러와 담고, mapping 이 되어 있지 않으면 translator 함수 호출

```python
col = 'data'
 
translator = Translator()                                                           # translator 함수 호출
 
df['eng'] = df[col]
try:
  for index, row in df.iterrows():
    query = row[col]
    if query in cache:
      df['eng'][index] = cache[query]
      continue
    else:
      response = translator.translate(query).text
      df['eng'][index] = response
      cache[query] = response                                                       # excel 에 추가 하기 위해서 cache dictionary 에 먼저 추가
      #if index % 10 == 0:                                                          # log 확인 위해서 10 배수마다 프린트
      #  print(index)
except AttributeError as e:
  print(index, row[col], e)                                                         # exception error 확인
finally:                                                                            
  print('updating cache')
  l_from, l_to = [], []
  for kor, eng in cache.items():                                                    # cache dictionary 에 있는 아이템들을 하나한 돌면서
    l_from.append(kor)                                                              # list 로 펴주고
    l_to.append(eng)
  cache_excel = pd.DataFrame({'from': l_from, 'to': l_to})                          # cache_excel file 덮어쓰기
  cache_excel.to_excel(cache_path)
```

## **4. Concat 해서 하나의 column 으로 합치기**

```python
tmp1 = df.drop(['eng'], axis=1)                                                   # 한영 변환된 칼럼인 'eng' 을 drop 한 temporary dataframe 만들기
tmp2 = df.drop([col], axis=1)                                                     # 한영 변환 대상 칼럼인 한글 칼럼을 drop 한 temporary dataframe 만들기
tmp2.rename(columns={'eng':'data'}, inplace=True)                                 # 칼럼명 통일
result = pd.concat([tmp1,tmp2], ignore_index=True)                                # 두 df 합치기
result = result.drop_duplicates()
```


## **5. 테스트 결과**

- **기존 데이터프레임 하단에 한영 변환된 data 값들이 append 되어 새로운 20개 행을 가진 데이터프레임을 리턴합니다**

![img](/assets/img/2021-09-10-googletrans-2/result.png)