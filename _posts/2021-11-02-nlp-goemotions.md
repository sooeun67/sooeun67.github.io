---
layout: single
title:  "[NLP 소식] 구글 AI에서 공개한 대규모 이모지 오픈소스 데이터셋 <GoEmotions>"
categories:
  - data science
tags:
  - NLP
  - Sentiment Analysis
  - 데이터분석
---

며칠 전, 구글 AI 에서 세분화된 27가지의 감정 분류된 데이터셋을 오픈소스로 공개했다. 

아래는 [구글 AI 블로그 내용](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html?fbclid=IwAR3EGJ_897cGdBxqMyAv34tjpYlGIxNufPUtlFs_0KwbzN-5csWcS-hm2l8)을 내가 이해하며 쓴 의/번역 및 요약 내용이다.

이모지(이모티콘)은 사회적 상호 작용이나 사람들간의 행동이나 관계 형성 하는 데에 있어 가장 핵심적인 측면이라 할 수 있다. 몇 가지 단어들로 다양한 범주의 미묘하고 복잡한 감정들을 표현할 수 있기 때문이다. 

지난 10년간, NLP 리서치 커뮤니티는 감성분류에 기반한 데이터셋을 만들어왔다. 대다수는 뉴스 헤드라인, 영화 자막, 우화 같은 영역들이고, 기본적인 6가지 감정(분노, 놀람, 혐오, 기쁨, 두려움, 슬픔) 에 초점이 맞춰져 있었다. 이 데이터들 덕분에 감성 분류에 대한 초기 탐색은 이루어질 수 있었지만, 대규모 데이터셋에 대한 필요성과 좀더 다양한 감정들에 대한 커버에 대한 논의가 나오기 시작했다.

[Paper](https://arxiv.org/pdf/2005.00547.pdf) 에서 레딧 사이트의 58,000개의 댓글들에 대해 27가지의 감정 카테고리로 레이블링/태깅 하였다. 긍정 감정은 기쁨 1가지만 포함하고 있는 기존 6가지 감정들과 다르게, GoEmotions에서는 12가지의 긍정적인 감정, 11가지 부정적인 감정, 4가지 모호한 감정, 1가지 중립적인 감정을 포함하고 있어 이모지 표현 간에 미묘한 차이가 있는 대화들을 좀더 잘 이해할 수 있게 되었다.

[튜토리얼](https://colab.research.google.com/github/tensorflow/models/blob/master/research/seq_flow_lite/demo/colab/emotion_colab.ipynb#scrollTo=UAz-tdQfuVBn)은 Tensorflow Model Garden 이라는 SOTA model implementation 모음집(?)에 올려놓아두었다고 한다. 

![goemotion1](/assets/img/2021-11-02-nlp-goemotions/goemotions-1.png)

방대한 양의 유저와 유저간의 대화들을 제공하는 래닷 플랫폼을 이용하여 초기 레딧 시기인 2005년 부터 2019년 1월까지의 레딧 사이트의 댓글들로 데이터셋을 만들었다(English only). 젊은 남성층이 많은 레딧 사용자를 감안하여 데이터 큐레이션을 적용하여 특정 층에 bias 가 생기지 않도록 하였고, 공격적인 댓글들을 미리 정의해놓은 terms 들로 인식하게끔 하였다. 신상이나 종교에 관한 concern 을 위해 데이터 필터링과 마스킹을 적용했고, 텍스트 길이 제한 또한 적용했으며, subreddit (특정 주제의 커뮤니티) 의 인기도를 고려하여 데이터 balance 를 맞췄다.

아래 3가지 목적을 최대 달성하기 위해 분류 체계를 만들었다

(1) 레딧 데이터에서의 감정들을 가장 많이 커버
(2) 가장 다양하게 감정 표현 종류들을 다룰 수 있게 제공
(3) 감정들의 수를 제한하고 겹치는 걸 줄이는 것

 데이터 라벨링 단계에서 총 56개의 감정 카테고리를 고려했었다고 한다. 문장에서 detect 하기 어렵거나 다른 감정들과 비슷해서 판단하는 평가자(raters) 입장에서 서로 align 되지 않는 감정들을 제거하고, 평가자들이 자주 제안하고 데이터에서 잘 표현되는 감정들을 추가했다.

94%의 예시들이 적어도 두 명의 평가자들에게 적어도 1개의 같은 감정 레이블이라는 동의를 얻었다.

![goemotions-2](/assets/img/2021-11-02-nlp-goemotions/goemotions-2.png)

감정들이 균일하게 분포되어 있지 않았고, 긍정적인 감정들이 자주 등장하는 경향을 보여줌에 따라 더 다양한 감정 분류 체계를 세웠다고 한다.

![goemotions-3](/assets/img/2021-11-02-nlp-goemotions/goemotions-3.png)

감정 분류 체계를 검증 하기 위해, PPCA(Principal Preserved Component Analysis) 주성분 분석을 적용했을 때, 각각의 component 가 significant 하다는 결론을 지을 수 있었기 때문에, 각각의 감정들이 데아터의 고유 부분들을 잘 표현해주고 있다 라고 볼 수 있었다고 한다. 클러스터링을 통해서는 각각 긍정/부정 끼리의 감정들이 closely related 되어 있는 것을 확인할 수 있었다. 쉽게 말해, 한 평가자(rater)가 어떤 댓글을 보고 "아 이건 'excitement'야" 라고 labeling 했을 때, 또다른 평가자가 같은 댓글을 보고, "fear" 라고 하기 보다는 연관성이 있는 감정인 "joy" 라고 할 경향이 높다 라고 보면 된다. 

![goemotions-4](/assets/img/2021-11-02-nlp-goemotions/goemotions-4.png)

[트위터 태그를 가지고 감정 분류를 이룬 논문](https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html?fbclid=IwAR3EGJ_897cGdBxqMyAv34tjpYlGIxNufPUtlFs_0KwbzN-5csWcS-hm2l8#:~:text=emotion-related Twitter tags)도 있는데, 트위터에 사용되는 언어가 너무 다양하기 때문에 적용성에 있어 제한이 있다고 생각한다. 이모지들은 트위터 태그들보다 더 표준화 되어 있고 편차가 덜 크기 때문에 좀더 consistent 하다고 할 수 있다. 트위터 태그나 이모지 모두 감정들을 직접적으로 이해하는 데에 목표가 있는 것이 아니라, 대화 표현의 다양성에 초점을 맞춘다고 이해해야 한다. 아래 예시와 같이, 🙏는 감사를, 🎂는 축하의 의미를, 🎁는 말그대로 선물을 뜻하고 있다. 한 개의 이모지가 복잡한 감정을 모두 대변할 수 없기 때문에, 감정 그대로를 대변하는 게 아니라 표현의 다양성을 대변한다고 이해한다. 

![goemotions-5](/assets/img/2021-11-02-nlp-goemotions/goemotions-5.png)
