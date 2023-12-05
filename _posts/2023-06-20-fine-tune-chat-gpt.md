---

layout: single
title:  "[ChatGPT/AI] 나만의 ChatGPT 모델 만들기 (w/ Fine-Tune)"
categories:
  - data science
tags:
  - AI
  - NLP
  - ChatGPT
  - OpenAI
  - 데보션
---

# Encountered Problems & Solutions

## 1. Issue: ```The number of classes in file-vFB5CE9LrZcacVJzMJnCuiTw does not match the number of classes specified in the hyperparameters.```

```python
# Fine tune
import os
os.environ['OPENAI_API_KEY'] = 'sk-bGJLbJDbVse......'

!openai api fine_tunes.create -t "eng_news_prepared_train (1).jsonl" -v "eng_news_prepared_valid (1).jsonl" --compute_classification_metrics --classification_n_classes 3
```

위 셀을 실행하면 아래와 같이 에러 발생으로 job failed 된다.

```python
Upload progress: 100% 1.48k/1.48k [00:00<00:00, 792kit/s]
Uploaded file from eng_news_prepared_train (1).jsonl: file-mwsvrLf2bNTDedWBFEniNmZB
Upload progress: 100% 379/379 [00:00<00:00, 377kit/s]
Uploaded file from eng_news_prepared_valid (1).jsonl: file-vFB5CE9LrZcacVJzMJnCuiTw
Created fine-tune: ft-2hKBY10pDrY2FzCDVd0qRszD
Streaming events until fine-tuning is complete...

(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[2023-04-19 01:24:33] Created fine-tune: ft-2hKBY10pDrY2FzCDVd0qRszD
[2023-04-19 01:25:15] Fine-tune failed. Errors:
The number of classes in file-vFB5CE9LrZcacVJzMJnCuiTw does not match the number of classes specified in the hyperparameters.

Job failed. Please contact support@openai.com if you need assistance.
```

Train 이나 Valid file에 충분한 데이터가 없다는 뜻.

## 2. Solution: Add more data!!!! 데이터를 추가하면 해결됨