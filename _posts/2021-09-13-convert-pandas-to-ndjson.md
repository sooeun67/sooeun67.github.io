---
layout: single
title:  "[python] Pandas 데이터프레임 NDJSON 타입으로 변환"
categories:
  - data science
tags:
  - pandas
---

판다스 데이터프레임을 JSON 타입으로 변환하는 작업은 [official doc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html) 에도 쉽게 찾아볼 수 있듯이 많은 이들에게 익숙할 것이다
모델 산출물을 Azure Storage 에 연동하여 업로드 하는데, NDJSON 타입으로 업로드 해달라는 요청을 받아 찾아보았다



### **NDJSON이란?**

- Newline Delimited JSON Data 이라는 용어 자체로 설명 가능
- 아래 왼쪽이 JSON, 오른쪽이 NDJSON ([Reference](https://medium.com/@kandros/newline-delimited-json-is-awesome-8f6259ed4b4b))
- [블로그](https://medium.com/@kandros/newline-delimited-json-is-awesome-8f6259ed4b4b)를 읽어보니 NDJSON 타입이 새로운 데이터를 추가할 때 모든 파일을 읽지 않아도 된다는 장점이 있다고 한다

![img](https://blog.kakaocdn.net/dn/bkeNAA/btreKJQDJ6y/rgKkMKxbEvM0RhG3KkrxK0/img.png)  


### **Pandas Dataframe 을 NDJSON 타입으로 <span style="color:red">READ(읽기)</span> 하고 싶은 경우** 에는

ndjson 타입으로 먼저 데이터를 로드한 후, 판다스 데이터프레임으로 감싸서 읽어오면 된다

```python
    with open(os.path.join({FILE_PATH}, {ITEM_FILE}), encoding='utf-8-sig') as f:	# 한글 데이터인 경우 encoding 설정
        data = ndjson.load(f)														# ndjson 으로 읽고
    org_data = pd.DataFrame(data)													# dataframe 으로 감싸서 읽기
```


### **Pandas Dataframe 을 NDJSON 타입으로 <span style="color:red">WRITE(쓰기)</span> 싶은 경우** 에는

pandas 의 to_json 을 사용하면서 parameter 설정을 **orient='records'** 와 **lines=True** 로 하면 된다.

```python
    df.to_json({file_path}, orient='records', lines=True)
```