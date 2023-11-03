---
layout: single
title:  "[Market Analysis] 유통 산업 분석 및 노트"
categories:
  - data science
tags:
  - 데이터분석
  - 산업분석
---

어제자 기사 참고 - 

## 쿠팡

## SSG 

## References
- http://theviewers.co.kr/View.aspx?No=2629757
- https://www.getnews.co.kr/news/articleView.html?idxno=608425
- 
며칠 전, 구글 AI 에서 세분화된 27가지의 감정 분류된 데이터셋을 오픈소스로 공개했다. 

아래는 [구글 AI 블로그 내용](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html?fbclid=IwAR3EGJ_897cGdBxqMyAv34tjpYlGIxNufPUtlFs_0KwbzN-5csWcS-hm2l8)을 내가 이해하며 쓴 의/번역 및 요약 내용이다.

이모지(이모티콘)은 사회적 상호 작용이나 사람들간의 행동이나 관계 형성 하는 데에 있어 가장 핵심적인 측면이라 할 수 있다. 몇 가지 단어들로 다양한 범주의 미묘하고 복잡한 감정들을 표현할 수 있기 때문이다. 


(1) 레딧 데이터에서의 감정들을 가장 많이 커버
(2) 가장 다양하게 감정 표현 종류들을 다룰 수 있게 제공
(3) 감정들의 수를 제한하고 겹치는 걸 줄이는 것

 데이터 라벨링 단계에서 총 56개의 감정 카테고리를 고려했었다고 한다. 문장에서 detect 하기 어렵거나 다른 감정들과 비슷해서 판단하는 평가자(raters) 입장에서 서로 align 되지 않는 감정들을 제거하고, 평가자들이 자주 제안하고 데이터에서 잘 표현되는 감정들을 추가했다.

94%의 예시들이 적어도 두 명의 평가자들에게 적어도 1개의 같은 감정 레이블이라는 동의를 얻었다.

![goemotions-2](/assets/img/2021-11-02-nlp-goemotions/goemotions-2.png)

감정들이 균일하게 분포되어 있지 않았고, 긍정적인 감정들이 자주 등장하는 경향을 보여줌에 따라 더 다양한 감정 분류 체계를 세웠다고 한다.
