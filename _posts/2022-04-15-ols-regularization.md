---
title: "Ridge와 Lasso는 왜 작동하는걸까?"
categories:
  - data-analysis
tags:
  - 일반 데이터 분석론
  - 수리과학
toc: true
sidebar:
  nav: data
use_math: true
---

# 들어가기 앞서...

이번 포스팅을 이해하기 위해서는 `OLS`에 대한 기본 개념과 `벡터 미적분학 (Vector Calculus, 혹은 Multivariate Calculus)`에 대한 이해가 필요합니다. 특히, `Lagrangian Multiplier`에 대한 내용을 알고 계시면 이해가 더 편합니다.

# Regularization

간단히 말하면, 특정 parameter에 의해 예측값이 치우쳐지는 경우, 이를 방지하고자 규제를 하는 것

보통은 cost function에 parameter 각각의 크기를 계산함

$$
Cost(\beta)=L(\beta)+Reg(\beta)
$$

여기서 $L(\beta)$: likelihood estimator, $Reg(\beta)$: regularizarion term

## Ridge

단순히, Regularization이 L2 norm인 경우를 의미

$$
Cost(\beta)=\|e\|_2^2+\lambda\|\beta\|_2^2
$$

## Lasso

단순히, Regularization이 L1 norm인 경우를 의미

$$
Cost(\beta)=\|e\|_2^2+\lambda\|\beta\|_1^1
$$

# 왜 저런 term이 실제로 regularization 역할을 하는가?

**Recall: Lagrange Multiplier**

$$
\text{For }f: \mathbb{R}^3\rightarrow\mathbb{R}, g: \mathbb{R}^3\rightarrow\mathbb{R},\\ \text{the critical point for }f(x) \text{ when }g(x)\leq c\\\text{ exists on the stationary point of the curve } \\L(\lambda, x)=f(x)+\lambda g(x)
$$

여기서, 우리는

$$
L(\lambda, x):=Cost(\beta, \lambda)\\
f(x):= L(\beta)\\
g(x):=Reg(\beta)
$$

임을 확인할 수 있다.

특히, $Reg(\beta)=\|\beta\|_1^1$ (Lasso) 이라는 것은, 마름모 영역으로 parameter를 가두는 것이기 때문에 몇몇 경우에서는 $\beta_i=0$으로 될 수 있고, 변수 선택법으로도 사용될 여지가 있음.

1. 간단한 OLS 식의 implicit form
2. regularization term을 추가하는 이유
3. L1 norm, L2 norm ⇒ ridge, lasso resp

라그랑주 승수법

1. Statement
2. 기하적인 의미

# 결론

문제 의식

1. 많은 블로그에서 ridge, lasso를 직관적으로 이해할 때 어떤 이상한 원이랑 사각형 갖고옴
2. 그거 왜그럼 ㅅㅂ

논문을 열심히 읽자

# 관련 페이퍼

Regression Shrinkage and Selection via the Lasso (Tibshirani, 1995)
