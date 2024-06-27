# 최적의 대화 제공, Us 서비스입니다.

## 목차



## 프로젝트 흐름도
<img width="1203" alt="image" src="https://github.com/Uarth/Us/assets/87052350/bd4dd956-424c-4965-b4c6-dc42fb6efb97">

## 프로젝트 구조도
<img width="758" alt="image" src="https://github.com/Uarth/Us/assets/87052350/a10b8b78-f2bc-4a76-ac29-8f97fe3ad80a">
### 1. 유저가 UI에서 가입 및기 본 정보 입력 -> User DB
### 2. 유저가 UI에서 녹음 실행 -> FastAPI로 이동해서 Feature 추출, Feature DB서버에 저장
### 3. 매칭시, Feature들을 불러와 Rule Based로 만든 후, 둘의 음성 녹음을 기반으로 LLM Enhanced Explain


### 유저 화면 예시 
<img width="855" alt="image" src="https://github.com/Uarth/Us/assets/87052350/b5f94b1f-0764-48c5-85d3-d0e0df988e91">
<img width="859" alt="image" src="https://github.com/Uarth/Us/assets/87052350/39e7d613-c37b-4997-9b7d-858a4e8e0693">



# Us

<div align="center">
<h3>24-1 YBIGTA 컨퍼런스</h3>

<em> 최적의 대화 상대 제공, Us입니다. </em>

</div>

## 목차
- [문제 정의](#문제-정의)
- [선행 연구](#선행-연구)
- [세부 목표](#세부-목표)
- [접근 방법](#접근-방법)
- [결과 및 주요 기능](#결과-및-주요-기능)
- [팀 구성](#팀-구성)

## 문제 정의
*좋은 대화를 제공하기 위해 매칭 기능과 설명을 제공한다*
*우리에겐 쉼이 되어주는 커뮤니티는 얼마 없다. 자주 보는 사람들에게 마음을 터놓고 이야기를 하기 쉽지 않고, 판단 당할 것 같은, 가까우니까 더 하기 어려운 말들이 있다. 이를 언어적/비언어적 요소로 분리해 매칭 기능을 구현했다.*

## 선행 연구

*LLM을 이용한 추천 시스템의 큰 두갈래*
  1. LLM을 이용한 Recommendation System
  2. LLM으로 설명을 제공하는 Recommendation System

## 세부 목표

*단순한 유사도 접근보다, 표면적이지 않은 요소들, 본질에 접근하려고 노력했다.*
  1. 대화에 중요한 요소
     - 발화에서 나타난 텐션 : 텐션이 비슷한 사람끼리 대화가 잘 맞을거라는 가정.
     - 발화의 높낮이 : 주파수와 진폭 외의 피처들을 이용해 유저의 특징을 포착하려고 시도함.
  2. 추천된 이유의 설명
     - Cold Start의 문제점은 데이터의 양이 부족해 딥러닝 등 깊은 단계의 추천을 보여주기 어렵다는 것에 있다.
     - 따라서 LLM을 이용해 추천된 사용자 간의 사유 설명을 시도한다.  

## 접근 방법

1. **태스크** *(세부 목표를 달성하기 위한 구체적인 태스크)*
   - 프론트팀
     - Figma + React 구현
   - 모델팀
     - Rule Based Recommendation
       - 발화에서 나타난 텐션 : 텐션이 비슷한 사람끼리 대화가 잘 맞을거라는 가정.
       - 발화의 높낮이 : 주파수와 진폭 외의 피처들을 이용해 유저의 특징을 포착하려고 시도함.
     - Explainable Recommendation
       - 추천 사유를 LLM을 이용해 설명

3. **데이터셋** *(사용한 데이터셋, API 등)*
    - 직접 수집
        - 학회원님들께 감사드립니다! 

4. **모델링/아키텍쳐 등** *(프로젝트 특성 및 목표에 따라)*
    - (Models)
        - Rule Based Recommendation System
          - 
    - (Service Architecture)
        - (Description)

## 결과 및 주요 기능

*음성을 녹음한 후, 답변의 내용과 보이스 특성을 기반으로 사람을 매칭*

## 팀 구성

|이름|팀|역할|
|-|-|-|
|(이동렬)|DS|(이동렬)|
|(김대솔)|DS|(김대솔)|
|(임채림)|DE|(임채림)|
|(목종원)|DE|(목종원)|
|(김인영)|DA|()|
|(정지훈)|DA|()|
|(김지훈)|DS|(김지훈)|
