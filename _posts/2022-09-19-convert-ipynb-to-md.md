---
layout: single
title:  "[Jupyter] 주피터 노트북 마크다운으로 초간단 변환하기"
categories:
  - data science
tags:
  - python
author_profile: false
---

코랩에서 주피터 노트북을 마크다운 으로 변환 하려면 아래와 같이 원라인 코드로 가능


```shell
!jupyter nbconvert --to markdown mynotebook.ipynb
```



