---
layout: single
title:  "Github Pages 재등록하는 방법"
categories: Github
tag: [github, problem-solving]
use_math: false
---

<br>

# 1. 문제의 발단
&nbsp; 내 Github Page를 입맛에 맞게 수정하는 중 내가 원하는 사항이 약 1시간 가량 업데이트가 안되어서 Page 등록을 취소하고 다시 등록하면 원하는 내용이 나오겠지라는 생각을 하게 되었다. 하지만... 등록 취소는 무사히 되었으나 이후 등록 버튼 자체가 활성화가 안되었다! 레포지토리 자체를 삭제하고 다시 생성해도 내가 생성하는 것이 아닌 포크한 레포지토리라 그런가 계속 버튼이 활성화가 안되었다...
![](/images/20230307_1.png)

<br><br>

# 2. 다시 Github Page 등록하는 방법
&nbsp; 일단 네이버나 구글로 검색하였을때 나와 같은 경우가 별로 없는지 쉽게 찾을수는 없었으나 영어로 어찌저찌 검색해서 겨우 찾았다. 아래 순서대로 진행하면 된다. 

관련 링크 : <https://stackoverflow.com/questions/73593914/publish-again-unpublished-github-pages-project>

<br>

1. 상단 Action - New workflow
![](/images/20230307_2.png)

2. 검색창에 'static html' 검색하여 configure 클릭
![](/images/20230307_3.png)

3. 'static.yml' 파일을 지정한 경로에 넣으면 자동으로 실행됨 (안되면 Rerun All jobs 실행)
![](/images/20230307_4.png)

4. 다시 레포지토리 설정창 가면 정상적으로 Github Pages가 등록되어 있는지 확인!
![](/images/20230307_5.png)

<br><br>

# 3. 배울 점
1. 역시 구글링을 하려면 영어로 하자.
2. 위기가 있어야 배우는 점이 있다.