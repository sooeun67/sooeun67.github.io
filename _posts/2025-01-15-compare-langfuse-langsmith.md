---
layout: single
title:  "[LLM/Langchain] Langfuse vs Langsmith 비교평가"
categories:
  - data science
tags:
  - AI
  - LLM
  - RAG
---

안녕하세요! 2025년 첫 포스팅으로는 Langchain 관련 내용으로 돌아왔습니다 :) 

RAG라는 핫한 키워드가 뜨면서 저도 Langchain 라이브러리들을 공부하고 써보고 있는데요, 관련해서 다양한 서비스 및 라이브러리들이 등장하고 있어서 비교평가를 해보고자 합니다.
우선 Langchain 을 처음 공부하다 보면, Langchain 을 구성하는 다양한 서비스들이 있어서 어떤 서비스를 사용해야 할지 고민이 되기도 합니다.
Langchain 에서 제공하는 서비스에는 같은 ecosystem 이라고 볼 수 있는 Langgraph, Langsmith 등이 있습니다.

### \*따라서 이번 포스팅에서는 <span style="color: #254abd">Langsmith와 Langfuse가 무엇인지 살펴보고 두 서비스를 비교평가</span>한 내용을 다룹니다. 두 개의 툴 모두 LLM기반 애플리케이션의 개발, monitoring 및 optimization 를 돕는 도구로 보면 되는데, 각각의 특징과 차이점을 좀더 살펴보고자 합니다.

# [Langsmith](https://smith.langchain.com/) 이란?
---------

Langsmith는 Langchain팀에서 개발한 LLM의 Ops 툴이라고 보면 될 것 같아요. 유료로 전환된 Closed Source Product 이지만, Free Tier 로 월 5,000회는 무료로 사용이 가능한 것으로 안내되어 있기 때문에 개인적으로 테스트 해보기에는 충분히 가능한 수준이라고 생각됩니다. 
LLM 애플리케이션을 잘 만들고 운영하기 위한 모니터링 및 최적화 도구라고 표현할 수 있을 것 같은데, 아래와 같은 기능을 제공합니다.

- 

# [Langfuse](https://langfuse.com/) 란?
---------
Langfuse는 Open Source 라는 점, 그리고 LLM 위에 구축된 애플리케이션에 중점을 두고 있다는 게 Langsmith와 가장 큰 차이라고 볼 수 있습니다.
따라서 Langfuse는 애플리케이션을 모니터링하고 디버깅하는 데에 중점을 두고 있습니다.



# 마치며
-----



# Reference
---------

- [https://medium.com/@heyamit10/langsmith-vs-langfuse-ef3d493ea74e](https://medium.com/@heyamit10/langsmith-vs-langfuse-ef3d493ea74e)
- [https://langfuse.com/faq/all/langsmith-alternative](https://langfuse.com/faq/all/langsmith-alternative)
