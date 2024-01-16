---

layout: single
title:  "[에러 해결] Python urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]"
categories:
  - data analysis
tags:
  - 에러 해결
  - python
  - pandas
---

# Encountered Problems & Solutions

## 1. Issue: ```URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1002)>```

아래 셀과 같이 에러 

![error](/assets/img/2024-01-16-solve-ssl-certificate-error/error.png)

## 2. Solution: SSL 인증 모듈 사용
아래 모듈과 코드를 추가하고 다시 실행하면 정상 작동된다

```python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```

![solved](/assets/img/2024-01-16-solve-ssl-certificate-error/solved.png)

