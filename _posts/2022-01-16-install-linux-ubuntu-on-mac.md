---
layout: single
title:  "[Ubuntu] 맥북 프로(Mac OS)에 VirtualBox로 Linux(Ubuntu 18.04) 설치 세팅하기(초보/차근차근 따라하기)"
categories:
tags:
  - linux
  - virtualbox
---

모델 돌리는데 Linux 환경이 필요해져서 집에 있는 윈도우PC에서 할까 하다 용량이 더 넉넉한 맥북에 가상머신을 설치해서 리눅스 환경을 셋팅하기로 했다.
순서는 생각보다 간단하다. 
- virtualbox & ubuntu 다운로드
- 가상머신(virtualbox) 에 ubuntu 환경 셋팅




# 1. VirtualBox 설치
### 아래 링크에 들어가서 **OS X hosts** 를 눌러 설치 파일 다운로드 받는다

- [virtualbox download link](https://www.virtualbox.org/wiki/Downloads)
![virtualbox](/assets/img/2022-01-16-install-linux-ubuntu-on-mac/virtualbox.png)

> 설치 시 "설치에 실패했다"라는 메시지가 뜨면 맥북의 **환경설정 > 보안 및 개인정보 보호 > 일반** 에서 ***차단된 Oracle 시스템 소프트웨어***를 **허용** 해주면 설치가능 

설치 완료 후 실행하면 아래와 같이 보인다.
![virtualbox2](/assets/img/2022-01-16-install-linux-ubuntu-on-mac/virtual-box-2.png)

# 2. ubuntu 설치

### 아래 링크에 들어가서 **Ubuntu Desktop** 을 눌러 설치 파일 다운로드 받는다
- [ubutu-download-link](https://ubuntu.com/download/desktop)

나는 18.04 버전으로 설치했는데, 공식 사이트에서 제공하는 LTS 버전은 20.04 이니 20.04 설치해도 무방할듯!

# 3. VirtualBox에 Ubuntu 설치

VirtualBox 를 실행하고 **새로 만들기** 를 누른다.

![3](/assets/img/2022-01-16-install-linux-ubuntu-on-mac/3.png)

이름에는 우분투 버전명을 입력하고 계속을 클릭

![4](/assets/img/2022-01-16-install-linux-ubuntu-on-mac/4.png)



macOS 버전이 Catalina 이거나 그 이후 버전이라면 ```export SDKROOT=$(xcrun --show-sdk-path)```  도 실행한다. 그 이유는 루비에 사용되는 헤더들의 위치가 이전과 바뀌어서 Jekyll 설치가 실패된다고 한다. 따라서 shell configuration을 통해 경로를 다시 설정해준다 정도로 이해했다.

### Homebrew 로 Ruby 설치

**Jekyll** 은 **Ruby v2.5.0** 보다 높은 버전을 요구하기 때문에, `ruby -v`  을 통해 버전 체크 해보고,

최신 버전의 Ruby가 필요하다면 Homebrew 를 통해 설치한다

```shell
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ruby
brew install ruby
```







> Github 서버에 반영될 때까지 업데이트를 매번 수분 기다릴 필요없이, 로컬에서 작업한 파일 및 수정사항들을 바로바로 볼 수 있기 때문에, 효율적인 환경이 구축되었다..!

## References
