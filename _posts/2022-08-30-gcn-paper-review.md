---
layout: single
title:  "[GNN/GCN] GNN/GCN이란 "
categories:
  - data science
tags:
  - NLP
  - python
toc: true
---



<br/>

# 0. GNN(Graph Neural Network) 



<br/>

# 

------



## Graph 란?

- 점들과 그 점들을 잇는 선으로 이루어진 데이터 구조. 관계나 상호작용을 나타내는 데이터. 
- 예) SNS 상의 친구관계
- 일반적으로 G=(V,E)로 정의하며, V = a set of nodes / E = the edges between the nodes



## Graph 분석 하기 어려운 이유

- 2D/3D Euclidean space 에 매핑하기 쉬운 Image, time-series signals, waves data 와 달리 그래프 데이터는 Euclidean space 상에 없기 때문에 해석하기 어렵다
- 고정된 형태가 아님 -- 다르게 생긴 그래프여도 행렬이 같을 수 있다
- 시각화로 볼 때 쉽게 이해할 수 없다 -- 무수히 많은 점들로 이어져 있기 때문에



## 그런데도 Graph 를 사용하겠다고? 왜?

- 관계나 상호작용(relationships and interactions) 같은 추상적인 개념들을 다루기 적합. 사회적 문맥을 파악하는 데 좋다고 함
- 복잡한 문제를 단순하게 재구현 가능하다
- SNS 분석이 가장 대표적인 예



- GNN은 주로 연결관계와 이웃들의 상태를 이용하여 각 점의 상태를 업데이트(학습)하고 마지막 상태를 통해 예측 업무를 수행한다. 
- 일반적으로 마지막 상태를 ‘node embedding’이라고 부른다.

<br/>



## GCN

- Recurrent GNN 에서는 같은 recurrent layer를 계속 반복하며 각 노드의 hidden state 를 일정 step 만큼 업데이트했었는데, Convolutional GNN 에서는 이 recurrent layer 대신에 각 convolutional layer 를 사용한다는 점이 다르다
- **Recurrent Layer 는 단계마다 동일한 가중치를 가진 레이어를 사용하는 반면, Convolutional Layer 는 각 단계별로 다른 가중치를 사용한다는 점이다.**

GNN의 동작은 따라서 크게 두가지로 생각할 수 있습니다.

1. propagation step - 이웃노드들의 정보를 받아서 현재 자신 노드의 상태를 업데이트 함
2. output step - task 수행을 위해 노드 벡터에서 task output를 출력함 -- 예를 들어, node label classification 이 task 였으면 node label 이 output





## Motivation : GNN ≈ CNN

다시 GNN으로 돌아오겠습니다. GNN의 아이디어는 Convolutional Neural Network(CNN)에서 시작되었습니다. CNN은 아래와 같은 특징을 가지고 있습니다.

- local connectivity
- shared weights
- use of Multi-layer

- 위와 같은 특징 때문에, CNN은 spatial feature를 계속해서 layer마다 계속해서 추출해 나가면서 고차원적인 특징을 표현할 수 있습니다. 위와 같은 특징은 마찬가지로 graph 영역에도 적용할 수 있습니다
- **Local Connectivity![2](/Users/sooeunoh/Documents/GitHub/sooeun67.github.io/assets/img/2022-08-30-gcn_paper_review/2.png)**

  <그림> 을 보면, CNN과 GNN의 유사한 점을 확인할 수 있습니다. 먼저, graph도 한 노드와 이웃노드 간의 관계를 local connectivity라 볼 수 있기 때문에, 한 노드의 특징을 뽑기 위해서 local connection에 있는 이웃노드들의 정보만 받아서 특징을 추출할 수 있습니다. 즉, CNN의 filter의 역할과 유사합니다.

  

  **Shared Weights**

  또한 이렇게 graph 노드의 특징을 추출하는 weight은 다른 노드의 특징을 추출하는데도 동일한 가중치를 사용할 수 있어(shared weight), computational cost를 줄일 수 있습니다.

  **Use of Multi-layer**

  CNN에서 multi layer 구조로 여러 레이어를 쌓게 되면 초반에는 low-level feature위주로 뽑고, 네트워크가 깊어질수록 high level feature를 뽑습니다. **graph같은 경우에 multi-layer구조로 쌓게되면 초반 layer는 단순히 이웃노드 간의 관계에 대해서만 특징을 추출하지만, 네트워크가 깊어질수록 나와 간접적으로 연결된 노드의 영향력까지 고려된 특징을 추출할 수 있게 됩니다.**

### Spatial-based Convolutional Graph Neural Networks

 

------

<br/>

# References

- https://medium.com/watcha/gnn-%EC%86%8C%EA%B0%9C-%EA%B8%B0%EC%B4%88%EB%B6%80%ED%84%B0-%EB%85%BC%EB%AC%B8%EA%B9%8C%EC%A7%80-96567b783479
- https://towardsdatascience.com/an-introduction-to-graph-neural-network-gnn-for-analysing-structured-data-afce79f4cfdc
- 
