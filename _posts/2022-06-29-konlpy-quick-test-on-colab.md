---
layout: single
title:  "[한국어 NLP] KoNLPy 패키지 코랩에서 쉽게 설치하고 사용하기"
categories:
  - data science
tags:
  - NLP
  - python
author_profile: false
---

### 한국어 자연어처리 패키지 (KoNLPy) 를 Colab 환경에서 쉽게 설치하고 사용해보자

## 설치

설치하는데 3분 정도는 걸리는 것 같다..!


```python
!curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash
```
<br/>

# 동작 확인 테스트

#### 형태소 분석기 마다 조금씩 다르게 분석 결과를 보여진다는 것을 확인 할 수 있음


```python
# KoNLPy의 Okt 와 Mecab 불러와서 사용해보기
from konlpy.tag import Okt, Mecab

Okt = Okt()
mecab = Mecab()

# 예시 문장
sentence = "OMG 공부하기 넘 싫네요. 그래도 스터디 해야됩니다 오마이갓! 지쟈스 하 아~"
```


```python
Okt.morphs(sentence)
```




    ['OMG',
     '공부',
     '하기',
     '넘',
     '싫네요',
     '.',
     '그래도',
     '스터디',
     '해야',
     '됩니다',
     '오마이갓',
     '!',
     '지쟈스',
     '하',
     '아',
     '~']




```python
mecab.morphs(sentence)
```




    ['OMG',
     '공부',
     '하',
     '기',
     '넘',
     '싫',
     '네요',
     '.',
     '그래도',
     '스터디',
     '해야',
     '됩니다',
     '오마이갓',
     '!',
     '지',
     '쟈',
     '스',
     '하',
     '아',
     '~']




```python
print("형태소 단위로 문장 분리")
print("----------------------")
print(Okt.morphs(sentence))
print(" ")
print("문장에서 명사 추출")
print("----------------------")
print(Okt.nouns(sentence))
print(" ")
print("품사 태킹(PoS)")
print("----------------------")
print(Okt.pos(sentence))
rst_list = Okt.morphs(sentence)
```

    형태소 단위로 문장 분리
    ----------------------
    ['OMG', '공부', '하기', '넘', '싫네요', '.', '그래도', '스터디', '해야', '됩니다', '오마이갓', '!', '지쟈스', '하', '아', '~']
     
    문장에서 명사 추출
    ----------------------
    ['공부', '스터디', '오마이갓', '지쟈스']
     
    품사 태킹(PoS)
    ----------------------
    [('OMG', 'Alpha'), ('공부', 'Noun'), ('하기', 'Verb'), ('넘', 'Verb'), ('싫네요', 'Adjective'), ('.', 'Punctuation'), ('그래도', 'Adverb'), ('스터디', 'Noun'), ('해야', 'Verb'), ('됩니다', 'Verb'), ('오마이갓', 'Noun'), ('!', 'Punctuation'), ('지쟈스', 'Noun'), ('하', 'Exclamation'), ('아', 'Exclamation'), ('~', 'Punctuation')]

<br/>

# 불용어 제거
#### `stop_words` 에 불용어를 추가 하여 처리


```python
stop_words = ". 을 를 하 아 ~ , 게 때 는 !"
stop_words = set(stop_words.split(' '))
word_tokens = Okt.morphs(sentence)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :',list(word_tokens))
print('불용어 제거 후 :',result)
```

    불용어 제거 전 : ['OMG', '공부', '하기', '넘', '싫네요', '.', '그래도', '스터디', '해야', '됩니다', '오마이갓', '!', '지쟈스', '하', '아', '~']
    불용어 제거 후 : ['OMG', '공부', '하기', '넘', '싫네요', '그래도', '스터디', '해야', '됩니다', '오마이갓', '지쟈스']

<br/>

# Optional) 불용어 파일로 불러와서 제거

#### `koreanStopwords.txt` 파일에 불용어를 쭉 추가한 다음, 이 파일을 불러와서 전처리 하는 방법


```python
f = open("/content/koreanStopwords.txt", 'r')
lines = f.readlines()
stopwords = []
for line in lines:
    line = line.replace('\n', '')
    stopwords.append(line)
f.close()
```


```python
print('불용어 단어 리스트: ', ' '.join(stopwords))
```

    불용어 단어 리스트:  이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤	 내 경우 명 생각 시간 그녀 다시 이런 앞 보이 번 나 다른 어떻 여자 개	 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 소리 놓	



```python
word_tokens = [token for token in word_tokens if token not in stopwords]
preprocessed_text= ' '.join(word_tokens)
print(preprocessed_text)
```

    OMG 공부 하기 넘 싫네요 . 그래도 스터디 해야 됩니다 오마이갓 ! 지쟈스 아 ~

<br/>

## 유의어/동의어 표준화 시키기
#### 비슷한 단어들을 표준화 하고 싶다면? 동의어 사전(단어사전)을 만들어 매핑시켜 **표준화** 하면 된다


```python
import pandas as pd 

synonym_dict = {
                '나' : ['저', '내'],
                '오마이갓' : ['OMG', 'Oh My God'],
                '공부' : ['스터디']
                }
                
apply_mapping = {word:k for k, v in synonym_dict.items() for word in v}
result_list = pd.DataFrame(rst_list).replace(apply_mapping, regex=True).values

print('유의어 처리 전 :',list(word_tokens))
print('유의어 처리 후:', list(result_list.flatten()))
```

    유의어 처리 전 : ['OMG', '공부', '하기', '넘', '싫네요', '.', '그래도', '스터디', '해야', '됩니다', '오마이갓', '!', '지쟈스', '아', '~']
    유의어 처리 후: ['오마이갓', '공부', '하기', '넘', '싫네요', '.', '그래도', '공부', '해야', '됩니다', '오마이갓', '!', '지쟈스', '하', '아', '~']

<br/>

## Reference

- `https://teddylee777.github.io/colab/colab-mecab`
