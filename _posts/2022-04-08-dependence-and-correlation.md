---
title: "Dependence와 Correlation"
categories:
  - data-analysis
tags:
  - 수리과학
  - 통계
toc: true
sidebar:
  nav: data
use_math: true
---

# 들어가기 앞서...

이번 내용을 이해하기 위해서는 `확률론(Probability Theory)`과 `선형 대수(Linear Algebra)`의 지식이 필요하며, `집합론(Set Theory)`에 대한 얕은 지식 역시 필요합니다.

# 두 변수가 관련이 없을때

일반적으로 두 변수가 관련이 없다는 것을 말할때는 두 변수가 서로 **독립적(independence)**이라고 말하거나, 서로 **연관되어있지 않다(uncorrelation)**고 이야기 합니다. 그러면 이것들은 수학적으로 어떻게 서술할 수 있고, 왜 그렇게 서술할 수 있으며, 무엇이 다른걸까요?

# 독립적인 변수의 정의

우선, independence부터 살펴봅시다. 우선 ‘변수’라는 단어를 정의하기 전에 사건 (event)이란 것을 정의해봅시다. 사건이란 가능한 경우를 의미합니다. 수학적으로 말하자면, 모든 경우를 원소로 가지고 있는 집합(표본 공간, sample space)의 부분 집합입니다. 예를 들면 주사위 2개를 던질 때의 표본 공간은 $\{(i,j): 1 \leq i, j ≤ 6\}$으로 정의할 수 있고, 첫 번째 주사위의 값이 1이 되는 사건의 집합은 전체 사건의 부분 집합 중 하나인 $\{(1,j): 1 \leq j ≤ 6\}$이 됩니다. (사건 집합이 무조건 유한할 필요는 없습니다.)

이제, 표본 공간을 $S$라고 정의하고 $A, B$를 사건 집합으로 둡시다. 이 때 두 사건 $A, B$가 서로 독립적이라는 말은 어떤 말을 의미할까요? 만약에 독립적이라면, $B$ 사건이 먼저 일어났든간에 $A$가 일어날 확률이 변하지 않아야겠죠? 반대로 $A$ 사건이 먼저 일어났는지 여부에 따라서 $B$의 확률이 바뀌면 안되겠구요. 이는 조건부 확률이랑도 관련이 있는데, 수학적인 statement를 쓰면 다음과 같이 독립성이 정의됩니다.

$$
[P(A|B)=P(A)]\ \&\ [P(B)=P(B|A)]
$$

다만, 조건부 확률은 일반적으로 다음과 같이 정의가 됩니다.

$$
P(A|B)=\frac{P(A\cap B)}{P(B)}
$$

따라서, 위의 정의를 조금 더 풀어헤치면 다음과 같이 바꿀 수 있습니다.

$$
\begin{align*}
P(A|B)&=P(A\cap B)/P(B)=P(A) \\
\Leftrightarrow& P(A)P(B)=P(A\cap B)
\end{align*}
$$

두 사건이 독립적이라는 것은 결국 두 사건이 동시에 일어날 확률이 각각이 일어날 확률의 곱과 같을때 부르는 말과 같군요! 뭔가 그럴듯합니다. 그러면 이제 변수에 대해서 독립성을 생각해봅시다. 일반적으로 변수는 여러 값을 가지고 있기 때문에, 변수 $X$에 대한 사건 집합은 보통 $f_X^{-1}(k)=\{x|f_X(x)=k\}$의 합으로 이루어집니다. 여기서 $f_X$는 특이한 말이 없는 이상 $X$의 확률 밀도 함수(probability density function)입니다. 그렇다면, 변수 $X$와 변수 $Y$가 서로 독립이라는 것은, 모든 변수의 값 $k_X, k_Y$에 대해서 $f_X^{-1}(k_X)$와 $f_Y^{-1}(k_Y)$가 독립이라는 것을 보이면 되겠군요.

# 두 변수간 관련이 있을 때

그렇다면 두 변수가 관련성이 없다는 것은 어떤 말일까요? 그 전에, 두 변수가 서로 관련이 있다는게 어떤 의미일지를 생각해봅시다. 변수 $X$가 움직일 때, 그에 맞춰 변수 $Y$가 움직인다면, 두 변수는 서로 관련이 있다고 볼 수 있겠네요. 여기서 같은 방향으로 움직이는지, 다른 방향으로 움직이는지는 중요하지 않습니다. 그저 **같이 움직인다**는 사실이 중요한 것이죠. 그렇다면 이걸 어떻게 수식으로 표현할 수 있을까요? $X$가 살짝 움직였다고 했을 때 $Y$가 평균적으로 얼만큼 움직였는지 파악함으로써 관련성을 볼 수 있지 않을까요? 비슷하게, $Y$가 살짝 움직였을 때 $X$의 평균 이동 거리를 볼 수도 있겠네요. 그러면 조금 더 엄밀하게 이 개념을 정의해봅시다.

> 관련성은 $X$가 평균값에서 $n$만큼 떨어졌을 때 $Y$가 평균값에서 멀어지는 것의 평균과
> $Y$가 평균값에서 $n$만큼 떨어졌을 때 $X$가 평균값에서 멀어지는 것의 평균의 곱이다

이를 수학적 statement로 쓰면 다음과 같습니다. 편의상 $cor_n$이라고 정의하겠습니다.

$$
cor_n(X, Y)=\mathbb{E}[Y-\mu_Y|X=\mu_X + n]\times\mathbb{E}[X-\mu_X|Y=\mu_Y + n]
$$

하지만 저희는 전체적인 추세를 보고자 하기 때문에, 또다시 이 값들을 평균을 취해서 $Y$가 벗어나는 기댓값을 구할 수 있습니다.

$$
\begin{align*}
cor(X, Y)&=\mathbb{E}[cor_n(X, Y)] \\
&=\mathbb{E}[\mathbb{E}[Y-\mu_Y|X=\mu_X + n]\mathbb{E}[X-\mu_X|Y=\mu_Y + n]]
\end{align*}
$$

그러면, law of iterated expectation을 통해서 다음과 같이 식을 수정할 수 있습니다.

$$
\mathbb{E}[(Y-\mu_Y)(X-\mu_X)]
$$

- **유도 과정 자세히 보기**
  우선, 일반적으로 law of iterated expectation의 정의는 다음과 같습니다. 이는 이번 포스팅에서 증명은 하지 않을 예정입니다.
  $$
  \mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]
  $$
  그렇다면, 두개의 랜덤변수가 곱해져있는 형태의 기댓값을 생각해보겠습니다.
  $$
  \mathbb{E}[XY]=\mathbb{E}[\mathbb{E}[XY|X=n]]=\mathbb{E}[X\times \mathbb{E}[Y|X=n]]
  $$
  여기서 $\mathbb{E}[Y|X=n]$은 이미 랜덤변수 $X$에 대한 함수이며, $Y$에 대해서 상수임을 쉽게 알 수 있습니다. 그렇다면
  $$
  \mathbb{E}[X\times \mathbb{E}[Y|X=n]]=\mathbb{E}[\mathbb{E}[X\times \mathbb{E}[Y|X=n]|Y=n]]
  $$
  으로 둘 수 있고, 이는 곧
  $$
  \mathbb{E}[\mathbb{E}[X\times \mathbb{E}[Y|X=n]|Y=n]] =
  \mathbb{E}[\mathbb{E}[X |Y=n]\mathbb{E}[Y|X=n]]
  $$
  으로 쓸 수 있습니다. 이를 통해 위의 유도를 끝낼 수 있습니다.

이것은 저희가 잘 알고 있는 $\operatorname{Cov}(X, Y)$와 같습니다. 또한 $\operatorname{Cov}(X, X)=Var(X)$임을 쉽게 알 수 있습니다. 중요한 점은, $\operatorname{Cov}(X, Y)$의 값이 한계점이 없이 왔다갔다 할 수 있습니다. 전문용어로는 unbounded되어있다, 혹은 유계가 아니다라고 이야기하는데, 이를 $[-1, 1]$ 인터벌 사이로 가두기 위해서 저희는 $X, Y$에 대한 standard deviation $\sigma_X, \sigma_Y$를 $\operatorname{Cov}(X, Y)$에 나눠줍니다.

$$
\operatorname{Corr}(X, Y)=\frac{\operatorname{Cov}(X, Y)}{\sigma_X\sigma_Y}
$$

이러면 저희는 쉽게 $\operatorname{Corr}(X, Y) \in [-1, 1]$에 있음을 알 수 있습니다.

- **Elegant한 증명**
  이는 일반적인 cauchy-schwatz thm을 알고 있어야 합니다.
  $(I, \langle \cdot, \cdot\rangle)$을 inner-product space로 정의합시다. 그렇다면 cauchy-schwatz thm이 의미하는 것은 다음과 같습니다.
  $$
  \forall x, y\in I, \langle x, x\rangle\langle y, y\rangle \geq \langle x, y\rangle^2
  $$
  여기서, inner product에 대한 공리는 3가지가 있습니다.
  1. $\langle x, y\rangle =\langle y, x\rangle$
  2. $\langle x, ay + bz\rangle=a\langle x, y\rangle+ b\langle x, z\rangle$ for $a, b\in F$: scalar
  3. $\langle x, x\rangle \geq 0$ and $=0$ only if $x=0$
     실제로, $\operatorname{Cov}(X, Y)$는 위의 세가지 조건을 모두 만족합니다. 따라서 $\operatorname{Cov}(\cdot, \cdot )$은 랜덤변수 공간의 inner product로서 정의가 가능하고, cauchy-schwatz thm에 의해서 자연스레
     $$
     \operatorname{Cov}(X, Y)^2\leq Var(X)Var(Y)
     $$
     를 만족하는걸 보였습니다. 따라서 $\operatorname{Corr}(X, Y) \in [-1, 1]$에 있음을 보였습니다.

그러면 다시 본론으로 넘어와서, 어떤 것들이 연관이 없다고 할까요? 바로, $X$가 움직이든 말든 $Y$는 평균적으로 변하지 않는다는 것을 의미하겠죠? 다른 말로 하자면

$$
\operatorname{Cov}(X, Y)=0
$$

이 uncorrelation의 정의가 될 수 있겠군요. 여기서 중요한 점은, $\operatorname{Cov}(\cdot, \cdot)$은 평균적인 추세를 잡아냅니다. 만약 $X$에 따라서 $Y$의 변동이 서로 다르게 작동하나 평균 변동값이 0이라면, covariance의 언어로 잡아낼 수 없습니다. 다시 말해, $\operatorname{Cov}(X, Y)=0$이 $X, Y$가 서로 independent한 것을 말해주지는 않는다는 것입니다!

# 결론

사실 예전부터 $\operatorname{Cov}$의 형태가 $\mathbb{E}[XY]-\mu_X\mu_Y$인 것이 항상 궁금했습니다. 그러다가 이번 기회에 왜 covariance가 이렇게 생겼는지에 대해 탐구를 해봤습니다. 많은 수학적인 개념들은 그냥 뜬금없이 생겨난건 없고, 항상 어떤 관찰과 직관, 그리고 문제의식 때문에 만들어졌다고 봐도 무방합니다. 이번 포스팅을 통해서 여러분도 covariance에 대한 새로운 직관이 생기셨으면 좋겠습니다.
