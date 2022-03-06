---
title: "Data Science Introduction"
categories:
  - data-analysis
tags:
  - 일반 데이터 분석론
toc: true
sidebar:
  nav: data
---

# 들어가기 앞서...

**데이터 분석은** 크게 세 퍼널로 이루어져있습니다.

`데이터 수집` ⇒ `데이터 전처리` ⇒ `모델링`

물론 데이터 수집과 전처리는 굉장히 중요한 파트입니다. 모 교수님에 의하면, 사실상 데이터 분석에서 가장 중요한 요소가 `데이터 전처리` 파트라고 할 만큼, 데이터 전처리에서 많은 시간과 노력이 들어갑니다. 좋은 데이터 셋이 있어야 분석이 용이하니까요.

하지만, 우리가 어떤 것을 분석하느냐에 따라서 데이터 전처리의 모양새가 달라질 수 있습니다. 또는, 데이터의 세팅이 달라질 수도 있습니다. 또한, 우리가 어떤 모델을 사용해서 분석을 하느냐 역시 중요합니다. 모든 모델이 같은 결과를 출력한다면 굉장히 그 가설이 강력한 경우이지만, 대개는 그렇게 못합니다. 따라서 그냥 무작정 몇가지 평가 지표를 통해서 모델을 선택하는 것이 아니라, 모델이 동작하는 원리를 이해하고 각 평가지표의 제대로 된 의미를 알고 있어야 질 좋은 데이터 분석을 할 수 있습니다.

이 세션에 적힌 글들은 앞으로 데이터 사이언스를 할 때 사람들이 간과하고 넘어갔던 부분들, 해석 방법 등에 대해 이야기합니다. 특히, ~~저의 출신이 수학과인 만큼,~~ 이 섹션의 글들은 대부분 선형 대수의 내용을 포함한 여러 수학적인 내용을 같이 담고 있습니다. 수학과 출신 혹은 그에 관심이 많으신 분들은 흡족하게 잘 이해할 수 있을 것이라고 예상합니다. 그래도 비전공자들을 위해 최대한 쉽게 쓰려고 노력하고 있기 때문에, 이해가 안되는 것들은 `kdchoi.mkt@gmail.com`으로 언제든지 메일을 주시면 제가 열심히 답변을 달거나 글을 수정을 하는 등 개선을 하겠습니다!