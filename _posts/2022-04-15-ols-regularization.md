---
title: "Ridge와 Lasso는 왜 작동하는걸까?"
categories:
  - data-analysis
tags:
  - 수리과학
  - 통계
  - 머신러닝
toc: true
sidebar:
  nav: data
use_math: true
---

# 들어가기 앞서...

이번 포스팅을 이해하기 위해서는 `OLS`에 대한 기본 개념과 `벡터 미적분학 (Vector Calculus, 혹은 Multivariate Calculus)`에 대한 이해가 필요합니다. 특히, `Lagrangian Multiplier`에 대한 내용을 알고 계시면 이해가 더 편합니다.

# Linear Regression과 OLS

지난번 OLS 섹션에서 이야기했던 것 처럼, Linear Regression을 통해서 각 설명변수가 어떻게 종속변수에 관여하는지, 그리고 그 의미가 정확하게 무엇인지에 대해 논의했습니다. 그러나, 몇몇 linear regression을 시행했을 때, 특정 변수에 해당하는 계수의 값이 지나치게 커져버려 오히려 예측력이 줄어들 수 있습니다.

예를 들어서, 우리가 다음과 같은 regression 결과를 얻었다고 해봅시다.

$$
Y = 20 + 100X_1 + X_2 + 3X_3
$$

이 경우, $X_1$의 값에 따라 $Y$의 값이 크게 바뀌게 되고, 이 경우 outlier의 예측에 대해 더 취약해지는, 불안정한 결과를 초래할 수 있습니다. 그렇기 때문에 통계학자들은 이러한 결과들을 보정하기 위해 계수에 대한 `penalty term`을 집어넣기 시작합니다. 이를 Regularization이라고도 부릅니다.

# Regularization

앞서 말했던 문제들과 같이 특정 parameter에 의해 예측값이 치우쳐지는 경우가 있을 수 있는데, 이를 방지하고자 `penalty term`을 추가해서 계수가 크게 늘어나는 것을 방지하는 테크닉을 `Regularization`이라고 이야기를 합니다. 이러한 penalty term은 어디에 추가를 해야할까요? Penalty term을 추가하는 이유가 계수 추정에 규제를 두는 것이기 때문에, 계수를 추정하는 식인 `Cost Function`에 추가를 합니다. 이를 수식적으로 풀어쓰면 다음과 같이 Cost Function이 재정의 될 수 있습니다.

$$
Cost(\beta)=L(\beta)+Reg(\beta)
$$

여기서 $L(\beta)$: likelihood estimator, $Reg(\beta)$: penalty term을 의미합니다. 여기에서 $Reg(\beta)$를 어떻게 정의하느냐에 따라 추정 결과가 달라질텐데, 일반적으로는 $L_1$ norm, 혹은 $L_2$ norm을 사용합니다. 이를 각각 `Lasso`와 `Ridge`라고 부릅니다.

## Lasso

단순히, Regularization이 L1 norm인 경우를 의미합니다

$$
Cost(\beta, \lambda)=\|e\|_2^2+\lambda\|\beta\|_1^1
$$

$e = Y - X\beta$ 를 의미하며, $\lambda$ 는 일반적으로 연구자가 지정하는 임의의 숫자입니다.
또한 $$\|\beta\|_1^1 = \sum_{k=1}^K |\beta_k|$$ 를 의미합니다.

## Ridge

단순히, Regularization이 L2 norm인 경우를 의미합니다.

$$
Cost(\beta, \lambda)=\|e\|_2^2+\lambda\|\beta\|_2^2
$$

$e = Y - X\beta$ 를 의미하며, $\lambda$는 일반적으로 연구자가 지정하는 임의의 숫자입니다.
또한 $$\|\beta\|_2^2 = \sum_{k=1}^K \beta_k^2$$ 를 의미합니다.

# 왜 저런 term이 실제로 regularization 역할을 하는가?

사실 이번 포스팅의 핵심 논의입니다. `Cost Function`에 어떤 `Regularization Term`을 추가했을 때 뭔가 베타값이 잘 규제가 될 것 같은 느낌이 듭니다. 그러나 정확한 매커니즘이 무엇일까요? 사실 많은 블로그에서는 Ridge나 Lasso를 설명하는데 있어 뜬금없이 마름모꼴의 도형과 타원 도형들을 좌표평면 위에 그려넣으면서 **Ridge는 올가미다, Lasso는 변수를 선택할 때 쓴다** 이런식으로 이야기를 합니다. 그런데 이에 대한 상세한 이야기는 많이 하지는 않은 것 같습니다. 어떤 수학적 원리가 동작하기에 Ridge와 Lasso가 그런 도형처럼 동작하는걸까요? 사실 이를 이해하기란 쉽지 않습니다. 왜냐하면 Ridge와 Lasso가 제대로 동작하는지 알려면 Lagrange Multiplier라는 Theorem을 알아야하고, 이는 대학 Calculus II에서 보통 배우거든요. 많은 사람들이 Multivariate Calculus를 접하면서 Topology 등에 대한 생소한 개념으로 뚜드려 맞으면서 정작 가장 중요한 개념인 Lagrange Multiplier에 대해서는 간과하고 넘어가는 경우가 많습니다. 그러나, 이는 사실 어려운 개념은 아닙니다. 우선 명제를 써보면 다음과 같습니다.

## Recall: Lagrange Multiplier

$$
\begin{align*}
&\text{For }f: \mathbb{R}^3\rightarrow\mathbb{R}: \text{differentiable ftn}, \\
&g: \mathbb{R}^3\rightarrow\mathbb{R},\\
&\text{the critical point for }f(x) \text{ when }g(x) = c\\
&\text{ exists on the stationary point of the curve } \\
&L(\lambda, x)=f(x)-\lambda g(x)
\end{align*}
$$

무수한 영어와 기호들이 악수를 요청하고 있지만, 이는 사실 어렵지 않습니다. 하나하나 보도록 하겠습니다.

1. 우선 $f$는 3차원 실수공간을 정의역으로 하고 실수를 치역으로 하는 미분 가능한 함수입니다. 공간점 하나를 넣으면 숫자 하나가 튀어나오는거죠
2. $g$ 역시 3차원 실수공간을 정의역으로 하고 실수를 치역으로 하는 함수입니다.
3. 이 경우에, 우리가 $g(x, y, z) = c$인 $x, y, z$를 가지고 있다고 해봅시다. $g = c$로 정의역을 constraint 한겁니다.
4. 그러면 이 상황에서 $f$의 critical point (극점)은 $L(x, y, z, \lambda) = f(x, y, z) - \lambda g(x, y, z)$로 정의된 $L$ function의 stationary point와 같습니다. (`Lagrange Multiplier`)

이를 그림으로 그리면 다음과 같은 상황을 의미합니다.

$f$는 각 점의 원점에서부터의 거리를 의미하며, $g = c$라고 constraint 되어있는 level curve입니다. 그렇다면 여기서 풀어야하는 문제는 이 level curve에서부터까지 거리의 최솟값을 구하는 문제입니다. 이 때 lagrange multiplier를 통해서 최소 거리를 갖는 점과 실제 거리를 계산할 수 있는 것입니다.

<p align="center">
  <img src="/assets/images/post_regularization_1.png" width="400px">
</p>
<p align = "center">
Fig.1 - Lagrange Multiplier의 예시
</p>

여기서 $\lambda$는 constraint value $c$의 값에 종속됩니다. 또한 여기서 서술된 Lagrange Multiplier는 3차원 공간에 대해 주로 이야기하는 Calculus II를 기반으로 서술되었으며, 일반적인 실수공간 $\mathbb{R}^n$에서도 적용될 수 있습니다. 이 Theorem을 기반으로 우리는 기존의 Regularization Technique를 revisit 해보겠습니다.

## Revisit: Regulaization

Regularization은 다음과 같은 식을 가지고 있습니다.

$$
Cost(\beta, \lambda) = L(\beta) + \lambda Reg(\beta)
$$

그런데 이는 Lagrange Multiplier와 비슷한 꼴을 가지고 있습니다. 왜나하면 최종 cost function에 $\beta$와 $\lambda$ 두개가 모두 들어있기 때문입니다. 그래서 이를 Lagrange Multiplier의 세팅과 비슷하게 세팅할 수 있습니다.

1. $L(\lambda, x):=Cost(\beta, \lambda)$
2. $f(x):= L(\beta)$
3. $g(x):=Reg(\beta)$

아까 이야기한 것 처럼 $\lambda$는 연구자가 임의로 설정하는 숫자라고 했습니다. 그리고 Lagrange Multiplier에서 $\lambda$는 $g(x)$의 constraint $c$와도 대응합니다. 이는 비약이긴 하다만, 간단하게 생각해보면 $\lambda$를 설정하는 것 자체가 $\beta$ 공간에 대한 constraint $c$를 설정하는 것과 동일하다고 볼 수 있을 것입니다. 이 세팅에서는 $f(x) = L(\beta)$의 최솟값이 굳이 원점에 있을 필요는 없고, 어떤 좌표평면의 특정 점을 중심점으로 $f(x)$의 그래프가 그려질 것입니다. 그리고 각 $\beta_i$의 축 별로 scaling이 다르기 때문에, 타원형으로 level surface가 그려질 것입니다. 개념을 간략하게 이야기하기 위해, $\beta$ coefficient가 2개밖에 없는 상황으로 이야기해보겠습니다. 2D에서는 3D를 볼 수 없기 때문에, level curve를 계속 그려나가면서 좌표평면에 그래프를 그려보도록 하겠습니다.

이 intuition 및 간략한 세팅 등을 통해서 regularization에 대한 자세한 이야기를 해보도록 하겠습니다.

## Revisit: Lasso

앞서 말씀드렸듯이, Lasso는 $L_1$ norm을 regularized term으로 가지고 있는 녀석입니다. 따라서 이 때 $g(x) = Reg(\beta_1, \beta_2) = c$는 마름모꼴의 형태를 하고 있습니다.

<p align="center">
  <img src="/assets/images/post_regularization_2.png" width="400px">
</p>
<p align = "center">
Fig.2 - Lasso
</p>
이 마름모꼴을 접선으로 $L(\beta_1, \beta_2)$이 스쳐지나간다면, 그 접점이 Regularization을 만족한 점이 될 것입니다.

## Revisit: Ridge

앞서 말씀드렸듯이, Ridge는 $L_2$ norm을 regularized term으로 가지고 있는 녀석입니다. 따라서 이 때 $g(x) = Reg(\beta_1, \beta_2) = c$는 원형의 형태를 하고 있습니다.

<p align="center">
  <img src="/assets/images/post_regularization_3.png" width="400px">
</p>
<p align = "center">
Fig.3 - Ridge
</p>
이 원의 접선과 $L(\beta_1, \beta_2)$의 접선이 같아진다면, 그 접점이 Regularization을 만족한 점이 될 것입니다.

# 결론

이번 포스팅에서는 regularization의 기하학적인 의미를 같이 탐구해봤습니다. 사실 Cost Function의 수치해석적으로도 생각할 수 있습니다. Regularization Term을 집어넣음으로서 기존 cost ftn $L(x)$의 값을 줄임과 동시에 $\beta$값도 동시에 줄어들게 해야하기 때문에 어떠한 새로운 균형점 $\tilde{\beta}$를 찾을 수 있다는 식으로 해석해도 무방하거든요.

사실 이번 포스팅을 했던 가장 큰 이유는, 제가 혼자 Ridge와 Lasso에 대해 공부를 해야할 일이 있었습니다. 그러나 블로그 포스팅을 보면 밑도 끝도 없이 타원형 / 마름모형 그래프를 그려놓고 이게 ridge다, 이게 lasso다하는 등의 겉핥기식 내용이 많았습니다. 또한, lasso는 무조건 마름모 끝점에서 만날 수 밖에 없기 때문에 변수 선택법으로도 탁월하다는 식으로 이야기를 한 것들을 여러개 봤습니다. 사실 이해가 안갔습니다. **왜 굳이 끝점에서 만나야하지? 그리고 실제로 제가 그렸던 Fig.3을 보면 당연하게도 그렇지 않은 경우가 많을텐데, 왜 저렇게 말을 할까?** 하는 의문점이 많았습니다.

그래서 제가 했던 것은 논문을 찾아보는 것이었죠. 어렵게 lasso라는 개념을 처음 정립한 논문인 95년도 Tibshirani 논문을 읽으면서, 모든 것이 해소되었습니다. **항상 끝점에서 만나는게 아니라, 끝점에서 만날 확률이 비교적 ridge 등 다른 방법론보다 높아서 변수 선택법으로 사용할 여지가 있다!**라고 언급을 한거죠.

생각해보면 블로그 글들을 보고 편하게 공부하려고 했던 저의 잘못이라고도 생각을 합니다. 영어보다는 한국어가 더 편하니까요. 그러나 블로그 글이 생각보다 엄밀하지 않은 것들이 많았고, 그리고 잘못된 해석이나 잘못된 내용이 포함되어있을 수 있다는 의식을 가지게 되었습니다. 그리고 이게 제가 블로그를 시작하게 된 가장 큰 원동력 중 하나이기도 하죠! 물론, 제 블로그 역시 저 개인이 쓰는 것이기 때문에 언제든 잘못된 내용을 제가 포함할 수 있습니다. 그렇기에 만약 잘못된 내용이 있다면 언제든 이메일로 이야기해주세요!

# 관련 페이퍼

_Regression Shrinkage and Selection via the Lasso (Tibshirani, 1995)_
