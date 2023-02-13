---
title: "Instrument Variable"
categories:
  - econometric
tags:
  - 일반 데이터 분석론
  - 내생 변수
toc: true
sidebar:
  nav: econ
use_math: true
---

# 들어가기 앞서...

이번 내용을 이해하기 위해서는 `linear regression`에 대한 이해와 `econometric`, `실험 설계` 등에 대한 내용을 숙지해야합니다. 아마도?

# Identifying Causal Inference

> 관측 데이터에 한해서 어떻게 정확한 인과관계를 추정할 수 있을까?

인과관계를 측정하기 위해서는 일반적으로 실험이 동반되어야한다. 여기서 말하는 실험이란, 사람들을 랜덤하게 A그룹, B그룹으로 나눈 후 각 그룹에 대해 서로 다른 처치를 주는 것을 의미한다. 여기서 가장 중요한건 **랜덤**인데, 일반적으로 주어진 데이터는 실험 데이터가 아닌 관측 데이터이기 때문에 **인과관계를 제대로 확인할 수 없다**.

예를 들어보자. 온라인 교육 시장에서 쿠폰을 사용하지 않고 구매한 사람들이 쿠폰을 사용하여 구매한 사람보다 더 완강을 많이 하는지를 알고 싶다. 만약 쿠폰을 준 사람들과 쿠폰을 안 준 사람들이 모두 랜덤하게 배정이 되었다면 그냥 각 그룹의 완강률 평균을 측정하면 되겠지만, **쿠폰을 사용한 그룹과 사용안한 그룹은 본질적으로 다른 그룹이다**! 쿠폰을 사용한 그룹은 가격에 민감한 그룹이고, 쿠폰을 사용하지 않은 그룹은 가격에 둔감한 그룹이기 때문에, 유저가 직접 쿠폰을 사용할지 말지 선택을 하고, 이것은 우리가 흔히 말하는 **Selection Bias**가 걸려있다고 볼 수 있다.

사람들이 가격에 민감한지 아닌지, 그리고 완강할 능력이 충분히 되는지 안되는지를 파악하고 싶지만, 사실 그런 변수들은 우리가 통제할 수 없다. 따라서 이런 변수들이 생략되면서 **Omitted Variable Bias** (생략 변수 편의)가 발생하게 되어, 우리가 보고싶은 쿠폰 사용 여부가 완강에 미치는 영향을 제대로 측정할 수 없게 되는 것이다. 그렇다면, 이 상황에서 우리는 어떻게 쿠폰 사용 여부의 영향을 미치는 것을 확인할 수 있을까? 만약 쿠폰 자체를 랜덤하는게 사용하는게 아니라, **쿠폰을 사용하는 것과 연관이 되지만 랜덤하게 발생하는 일**들을 가지고 오게 된다면? 이렇게 랜덤하게 발생하는 일을 통해서 유저들을 처치군과 대조군으로 랜덤하게 나눌 수 있을텐데, 그러면 그 사람들에 대한 쿠폰 효과를 측정한다면 그게 적당한 처치효과로 볼 수 있지 않을까? 그러니까, 이런 상황을 잘 사용한다면 실험과 비슷한 세팅이 될 수 있지 않을까?

학계에서는 이렇게 랜덤하게 발생하는 사건 및 변수를 **Instrumental Variable** (도구 변수)라고 부른다. 어쨌든 회귀분석은 수학적인 식으로 구성되기 때문에, **instrumental variable**도 수학적으로 정의해보겠다.

# Definition of IV

우선, 우리가 보고 싶은 것은 $X$가 $y$에 얼마나 영향을 주는지를 알고싶다. 이는 다음과 같은 regression equation으로 볼 수 있다.

$$
y =X\beta+\epsilon
$$

다만, 여기서 생기는 문제는 $X$가 완전히 $y$를 통제하지 못한다는 점에 있다. 이 말인 즉슨, $\operatorname{Cov}(y, \epsilon)\neq 0$임을 의미한다. 이 상황에서, 우리는 어떻게든 저러한 error를 없애서 $y$와의 correlation을 없애버리고 싶다. 위에서 설명했지만, 이 상황을 타개할 수 있는 것은 instrumental variable, $Z$이다. 여기서 $Z$가 가져야하는 특성을 알아보자.

앞서 말했던 것과 비슷하게, instrumental variable은 우리가 보고 싶은 변수와 연관이 되면서, 랜덤하게 발생해야한다. 이를 수식적으로 풀어보면 두가지로 축약할 수 있는데

1. $\operatorname{Cov}(X, Z)\neq 0$
2. $\operatorname{Cov}(Z, \epsilon)=0$

이다. 특히, (2)번의 condition은 굉장히 중요해서 이름이 있는데, 바로 `**exclusion restriction**`이다. 사실 $Z$가 정말 pure random할 필요는 없고, 단순히 에러, 즉 $\epsilon$과 correlation만 없어도 된다. 그러나 pure random이면 pseudo random이기 때문에, 보통 $Z$를 exogenous shock으로 두기도 한다. 그러면 이걸로 어떻게 true coefficient를 측정할 수 있을까?

# Identification

현재 우리의 상황을 다시 한 번 돌아보자. $X$는 제대로 $y$를 컨트롤 못해서 $\epsilon$과 $y$가 서로 correlation이 있고, $Z$는 $X$와 관련이 있으면서 $\epsilon$과는 무관하다. 그렇다면, 우선 $Z$가 $X$에 미치는 영향을 알아보자. 이는 다음과 같은 regression으로 coefficient $\pi$를 측정할 수 있다.

$$
X=\pi Z+ \eta
$$

그렇다면, 우리가 구한 $\pi$의 추정치 $\hat{\pi}$를 통해서 $Z$가 $X$에 미치는 추정치인 $\hat{X}=\hat{\pi} Z$를 찾을 수 있다. 여기서 중요한 점은, 이제 $\hat{X}$는 random하게 발생한 $Z$에 따라 분포가 되었다는 것이다. 이 $\hat{X}$를 원래 우리의 모형 $y=X\beta + \epsilon$에 집어넣자.

$$
y = \hat{X}\beta + \epsilon
$$

이렇게 바뀌면, 이제 더이상 $\hat{X}$와 $\epsilon$의 correlation은 존재하지 않게 된다. 그 이유는,

$$
\operatorname{Cov}(\hat{X}, \epsilon) = \hat{\pi}\operatorname{Cov}(Z, \epsilon) = 0
$$

그런데 여기서, 우리가 $\hat{\beta}$를 구하게 되면

$$
\hat{\beta} = \frac{\operatorname{Cov}(\hat{X}, y)}{\operatorname{Var}(\hat{X})}
$$

가 되며, $\hat{X} = \hat{\pi}Z$이기 때문에,

$$
\hat{\beta} = \frac{\hat{\pi}\operatorname{Cov}(Z, y)}{\hat{\pi}^2\operatorname{Var}(Z)} = \frac{\operatorname{Cov}(Z, y)}{\hat{\pi}\operatorname{Var}(Z)}
$$

이고, 사실상 $\hat{\pi} = \operatorname{Cov}(Z, X)/\operatorname{Var}(Z)$이기 때문에,

$$
\hat{\beta} = \frac{\operatorname{Cov}(Z, y)}{\operatorname{Cov}(Z, X)}
$$

로 측정할 수 있다. 특히, 잘 보면 저것들은 $y$에 대한 $Z$의 회귀식 계수값을 $X$에 대한 $Z$의 회귀식 계수값으로 나눈 값이기 때문에, 단순히 두 식을 regression한 후에 계수만 나누면 true beta 값을 찾을 수 있다는 것이다.

# Wald Estimator - Z가 discrete할 때

이러한 pseudo-random을 가진 instrumental variable Z가 만약 binary 변수라면? 그렇다면 우리가 구한 beta값은 다시 써질 수 있다.

$$
\hat{\beta} = \frac{\operatorname{Cov}(Z, y)/\operatorname{Var(Z)}}{\operatorname{Cov}(Z, X)/\operatorname{Var(Z)}} = \frac{\mathbb{E}[y|Z=1]-\mathbb{E}[y|Z=0]}{\mathbb{E}[X|Z=1]-\mathbb{E}[X|Z=0]}
$$

이 이유는 간단한데, 일반 regression 식 자체가 사실은 conditional mean을 다루기 때문이다. 즉

$$
y=\alpha + Z\rho + \epsilon \\
\implies \mathbb{E}[y|Z=1] - \mathbb{E}[y|Z=0] = \rho
$$

가 되기 때문에, 굳이 regression coefficient를 추정하지 않고서도 단순 statistic을 통해 causal inference를 계산할 수 있는 것이다. 저런 형태의 estimator를 특히 **_Wald Estimator_**라고 부른다.

# 결론

Instrumental variable의 가장 큰 특징은 $y$에 대해서 $X$와 관련있는 정보만 보존하고 $X$와 관련 없는 정보는 모두 없애버린다는 것이다. 비슷하게, instrumental variable을 여러개 사용해도 문제 없다. 그렇다면, 엄청 많은 IV를 사용하는게 항상 좋을까? 그리고 모든 instrumental variable이 다 좋을까? 여기에서 두가지 문제점이 있는데

1. Overidentification: 많은 개수의 IV를 사용함으로써 오히려 검정 효과가 낮아지는 현상
2. Weak Instrument: Exclusion Restriction이 제대로 working하지 않아서 omitted variable bias의 효과가 줄어들지 않는 현상

등의 문제가 있으므로, 잘 확인하고 사용해야한다.

이 토픽에서 계속 다루었던 것은 처치효과가 모든 사람들에게 똑같은 경우에 대해서만 다루었다. 그러나, 항상 이런 효과가 똑같을 이유는 없다. 만약 처치 효과가 각 사람들마다 다르다면 이러한 $\rho$값을 어떻게 해석해야할까? 이는 다음 토픽인 [Econometric 2. LATE Theorem](https://www.notion.so/Econometric-2-LATE-Theorem-ca084b2f12754015a33be8f575f07cef) 으로 넘기도록 하겠다.

# 관련 페이퍼

Angrist and Pisckhe (2008) - _Mostly Harmless Econometrics: An Empiricist's Companion_

Goldfarb and Tucker (2014) - _Conducting Research with Quasi-Experiments: A Guide for Marketers_
