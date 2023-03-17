---
layout: single
title:  "Jekyll로 Github 블로그 연동하여 개발 환경 구축하기(Mac OS)"
categories:
  - Github
tags:
  - github
  - blog
  - jekyll
toc: true
---



Github Blog 를 운영할 때, 업데이트 사항을 바로바로 보기는 힘들고 짧으면 1분 내외, 길면 3-4분까지 서버 호스팅에 소요된다. 따라서 로컬에서 업데이트 내용을 바로바로 확인할 수 있는 개발 환경을 구축했던 단계를 정리해보겠다. 

## Jekyll 

Jekyll is a static site generator

Jekyll의 [official doc](https://jekyllrb.com/docs/) 의 Quickstart 와 [Prerequisites](https://jekyllrb.com/docs/installation/macos/) 를 기본으로 따라가며 진행하였다.

- **개발 환경** 은 MacOS



# 1. Prerequisites 설치

### Command Line Tools 설치

터미널을 열고 command line tool 먼저 설치한다

```shell
xcode-select --install
```



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

해보니까 시간이 2분 이상 소요되었다. 조금 걸린다.

```shell
# 어떤 shell 이용하는지 확인해보고 Zsh/Bash 타입에 따라 아래 둘 중 하나의 코드를 실행
echo $SHELL

# Zsh 의 경우
echo 'export PATH="/usr/local/opt/ruby/bin:/usr/local/lib/ruby/gems/3.0.0/bin:$PATH"' >> ~/.zshrc

# Bash 의 경우
echo 'export PATH="/usr/local/opt/ruby/bin:/usr/local/lib/ruby/gems/3.0.0/bin:$PATH"' >> ~/.bash_profile
```



Terminal 다시 켜서 루비 버전 확인해보자

```shell
which ruby
# /usr/local/opt/ruby/bin/ruby

ruby -v
#ruby 3.0.3p157 (2021-11-24 revision 3fb7d2cadc) [x86_64-darwin20]
```



Official doc 에 보면 **rbenv** 설치도 설명해주는데, 다양한 루비 버전을 사용할 때 유용하다고 해서, 나는 skip!

### Jekyll 설치 

이제 bundler 와 jekyll gems 를 설치해준다

```shell
gem install --user-install bundler jekyll
```

다시 루비 버전 확인하고, 루비 버전의 앞 두 자리수 (나의 경우는 3과 0) `X.X` 자리에 넣어준다

```shell
ruby -v
#ruby 3.0.3p157 (2021-11-24 revision 3fb7d2cadc) [x86_64-darwin20]
```

```shell
# Zsh 의 경우
echo 'export PATH="$HOME/.gem/ruby/X.X.0/bin:$PATH"' >> ~/.zshrc
# 내 경우는: echo 'export PATH="$HOME/.gem/ruby/3.0.0/bin:$PATH"' >> ~/.zshrc

# Bash 의 경우
echo 'export PATH="$HOME/.gem/ruby/X.X.0/bin:$PATH"' >> ~/.bash_profile
```



이제 prerequisite 끝! 마지막으로 home directory로 간다. 아래와 같이 summary 정보가 나온다 

```shell
gem env
```


![cmd](/assets/img/set-up-github-page-with-jekyll-command-line.png)


# 2. Jekyll 로 로컬 서버 실행

## Jekyll 설치

이제 필수 준비 단계는 끝났다. Gem 을 통해 Jekyll을 설치해보자

```shell
gem install jekyll bundler
```



## Github Blog 폴더로 가기

터미널 에서 나의 Github Blog(Page) 를 연다. 페이지 폴더들이 있는 것을 확인할 수 있다

```shell
(base) sooeunoh@MacBook-Pro sooeun67.github.io % ls
CHANGELOG.md			banner.js
Gemfile				docs
LICENSE				images
README.md			index.html
Rakefile			minimal-mistakes-jekyll.gemspec
_config.yml			package-lock.json
_data				package.json
_includes			screenshot-layouts.png
_layouts			screenshot.png
_posts				staticman.yml
_sass				test
assets
```

### github.io 페이지 폴더에 bundle 설치

```shell
bundle install
```

> 설치 시, ```/Users/sooeunoh/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve/servlet.rb:3:in `require': cannot load such file -- webrick (LoadError)``` 와 같은 에러가 떠서, `bundle add webrick` 을 실행하여 webrick 을 추가해줬더니 해결되었다



### 사이트 및 로컬 서버 구축

```shell
bundle exec jekyll serve

# 아래와 같이 server address 나올 것
(base) sooeunoh@MacBook-Pro sooeun67.github.io % bundle exec jekyll serve
Configuration file: /Users/sooeunoh/Documents/GitHub/sooeun67.github.io/_config.yml
            Source: /Users/sooeunoh/Documents/GitHub/sooeun67.github.io
       Destination: /Users/sooeunoh/Documents/GitHub/sooeun67.github.io/_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
       Jekyll Feed: Generating feed for posts
                    done in 1.292 seconds.
 Auto-regeneration: enabled for '/Users/sooeunoh/Documents/GitHub/sooeun67.github.io'
    Server address: http://127.0.0.1:4000
  Server running... press ctrl-c to stop.
```

**[http://localhost:4000/](http://localhost:4000/)** 로 접속하여 **Jekyll** 사이트가 정상적으로 **동작**하는 지 확인할 수 있다.





> Github 서버에 반영될 때까지 업데이트를 매번 수분 기다릴 필요없이, 로컬에서 작업한 파일 및 수정사항들을 바로바로 볼 수 있기 때문에, 효율적인 환경이 구축되었다..!
