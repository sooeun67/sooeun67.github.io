---
layout: single
title:  "[Github] 초보자를 위한 SSH 키 만들고 등록하기: Official Doc 따라해봅시다"
categories:
  - Github
tags:
  - blog
---

Github 에서 push/pull 하기 위해서는 SSH Key 가 필요합니다. 

Git 의 [Official Doc](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) 을 참고 하여 SSH 키를 생성하고 등록해봅시다. (굉장히 친절하게 나와있음..)

- Terminal 에서 진행


## **0. SSH Key 있는지 확인 하기**

키를 생성하기에 앞서, SSH Key 가 존재하는지 확인 해봅니다. 아래와 같은 key 파일들이 없으면 새로 생성해야 합니다. 

- *id_rsa.pub*
- *id_ecdsa.pub*
- *id_ed25519.pub*

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % ls -al ~/.ssh
total 8
drwx------   3 sooeunoh  staff    96  9 23  2020 .
drwxr-xr-x+ 56 sooeunoh  staff  1792  6 17 11:23 ..
-rw-r--r--   1 sooeunoh  staff   801 10 21  2020 known_hosts
```

 (저는 없기 때문에 다음 스텝으로..)


## **1. SSH Key 만들기**

터미널에 **ssh-keygen -t ed25519 -C "Github계정에서 사용하는 나의 이메일주소"** 만 치면, 알아서 key pair 가 생성됩니다.

비밀번호를 설정하고 싶다면, ***passphrase*** 에 비밀번호를 입력하고 한 번 더 입력합니다. 

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % ssh-keygen -t ed25519 -C "Github계정에서 사용하는 나의 이메일주소"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/Users/sooeunoh/.ssh/id_ed25519): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/sooeunoh/.ssh/id_ed25519.
Your public key has been saved in /Users/sooeunoh/.ssh/id_ed25519.pub.
```

## **2. SSH Key 등록하기**

이제 1단계에서 생성한 SSH Key를 SSH Agent 에 등록합니다.

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % eval "$(ssh-agent -s)"
Agent pid 99680
```

맥 버전이 macOS Sierra 10.12.2 이거나 더 최신 버전이라면, config 파일을 수정해야 합니다. 

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % open ~/.ssh/config			# config 파일 열기
The file /Users/sooeunoh/.ssh/config does not exist.						# 없다고 나옴
(base) sooeunoh@MacBook-Pro sooeun67.github.io % touch ~/.ssh/config		# config 파일 생성
(base) sooeunoh@MacBook-Pro sooeun67.github.io % open ~/.ssh/config			# config 파일 열기
```

config 파일이 없다면 위와 같이 만들고, 엽니다. 빈 메모장 파일이 열릴 거에요. 거기에 아래 정보를 넣고 저장합니다.

```shell
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

빈 메모장 파일이 열리면 이 정보를 입력하고 저장합니다
![img1](/assets/img/2021-06-17-github-ssh-key/config.png)


이제 Private SSH Key 를 등록합니다. 아까 생성할 때 비밀번호를 설정했다면, 비밀번호도 입력해주세요.

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % ssh-add -K ~/.ssh/id_ed25519
Enter passphrase for /Users/sooeunoh/.ssh/id_ed25519: 
Identity added: /Users/sooeunoh/.ssh/id_ed25519 ("나의 이메일")
```

 

## **3. SSH Key 를 Github 에 등록하기**

자, 마지막 단계입니다!

![img2](/assets/img/2021-06-17-github-ssh-key/ssh-1.png)
![img3](/assets/img/2021-06-17-github-ssh-key/ssh-2.png)

Github 에 로그인하고, Settings (설정) -> SSH and GPG Keys 으로 갑니다.

New SSH Key 클릭


![img4](/assets/img/2021-06-17-github-ssh-key/ssh-3.png)New SSH Key 클릭

Title 은 적당히 써 주시면 되고(저는 macbook pro라고 했음), Key 칸에 생성한 키값을 복사해와서 붙일 겁니다. 

![img5](/assets/img/2021-06-17-github-ssh-key/ssh-4.png)

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % pbcopy < ~/.ssh/id_ed25519.pub
```

끝!!!!

### 등록후에 Github 에 이렇게 나올 겁니다!!!!


![img](https://blog.kakaocdn.net/dn/6I5wf/btq7wO6qc5Z/0qKQLtb7H4OWRv7fdOM7EK/img.png)


```
(base) sooeunoh@MacBook-Pro ~ % ssh -T git@github.com

Hi sooeun67! You've successfully authenticated, but GitHub does not provide shell access.
```

※ Reference

- https://mochadwi.medium.com/what-to-do-when-ssh-ing-permission-denied-publickey-2a1194188563