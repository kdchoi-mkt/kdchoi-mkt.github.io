---
title: "분류 예측 이해하기 - Logistic Regression"
categories:
  - data-analysis
tags:
  - 계량경제학
  - 통계
  - 머신러닝
toc: true
sidebar:
  nav: data
use_math: true
---

# 들어가기 앞서...

이번 내용을 이해하기 위해서는 `고등학교 수준의 수학 지식(미적분)`, 그리고 `확률론`과 `분포`에 관한 내용만 이해하고 있으면 충분합니다.

# 확률을 추정하는 모델

일반적으로 outcome variable을 예측하기 위해서는 linear regression을 기반으로 모델을 구성합니다. 그러나 만약 outcome variable이 0 또는 1만 있는 `binary` 변수라면? 일반 linear regression을 사용해서 binary 변수를 예측하고자 하면 값이 항상 0$\sim$1 사이에 나오는 것이 아니라 음수, 혹은 1보다 큰 “확률의 의미를 가질 수 없는 값”이 출력될 수 있습니다. 그러면 이를 해결하기 위해 어떻게 해야할까요? 가장 심플한 방법은 linear regression을 한 뒤에 이 **연속적인 값을 0$\sim$1로 가두어버리는 방법**입니다. 수학적으로는, 어떤 다음 성질을 만족하는 연속함수 $g(x)$를 가져와서 linear regression 결과를 $g(x)$로 한 번 더 convert하는 방법이 있을 것입니다.

$$
g(x): \mathbb{R}\to [0, 1], g(\alpha + \beta_1x_1 + \cdots + \beta_kx_k) \in [0, 1]
$$

그러면 우리의 outcome variable은 무조건 0과 1 사이에 가둬지고, 확률의 의미를 어느정도 가질 수 있게 되는 것이죠. 그렇다면 저렇게 실수 전체를 정의역으로 하고 0~1사이를 치역으로 하는 좋은 함수 $g(x)$는 어떤게 있을까요?

# 좋은 함수 탐색하기

어떤 성질을 가지는 $g(x)$가 좋은 함수라고 부를 수 있을까요? 우선, 당연히 연속이어야하고 미분가능해야할 것입니다. 그리고 만약 단조증가함수라면 더할 나위 없이 좋겠죠! 역함수가 존재한다는 것은 각 확률마다 inverse를 취할 수 있다는 뜻이기도 하기에, 이는 정보를 우리에게 손실없이 원래대로 제공해줄 수 있다는 뜻이기도 합니다. 그리고 이러한 $g(x)$는 우리가 알고 있는 다항함수나 지수, 로그 함수의 형태로 나타나면 좋을 것 같네요. 특히, $x$가 $-\infty$가 될수록 $g(x) = 0$이 되고, $x$가 $\infty$가 될 수록 $g(x) =1$이 되면 특히 더 좋을 것 같구요. 그러면 정리를 해봅시다

> 좋은 함수 $g(x)$의 성질

1. 연속이고 미분가능하다
2. 단조 증가함수이다 (역함수가 존재한다)
3. Elementary Function으로 이루어져있다
4. $\lim_{x\to-\infty}g(x) = 0, \lim_{x\to\infty}g(x) =1$이다.

이것들을 만족하는 친구들은 여러가지가 있겠으나, 그 중 우리가 가장 많이 쓰는 것은 `expit 형태의 무언가`입니다.

$$
\sigma(x):= \frac{e^x}{1 + e^x}
$$

이녀석은 연속이고 미분가능하며, 단조증가함수에, 지수함수의 꼴로 이루어져있고, 또한 우리가 원하는 limit의 형태를 그대로 가지고 있습니다. 그렇다면 linear regression 형태에 이녀석을 composite하면 확률의 의미를 가지는 outcome이 나올 것이고, 이를 통해 우리가 원하는 $\beta$값을 추정할 수 있을 것입니다. 물론 이 $\beta$는 우리가 원하는 어떠한 cost function의 $\argmin$ 값일 것이죠. 즉,

$$
\beta = \argmin_\beta \operatorname{Cost}(\sigma(\alpha + \beta_1x_{1i} + \cdots + \beta_kx_{ki}), Y_i)
$$

을 의미합니다. 특히 여기서 $\operatorname{Cost}$ 함수가 Bernoulli 분포의 MLE와 같다면, 우리는 이 회귀분석을 통해 $\beta$를 추정하는 방법론을 `logistic regression`이라고 부릅니다.

# Econometric에서의 Logistic Regression

그렇다면, 이 logistic regression은 단지 0과 1을 예측하는 회귀 분석일까요? 수학적으로 어떨때 데이터가 0을 가리키고 1을 가리킬까요? 이렇게 생각해봅시다. 우리가 어떤 제품을 선택하는데 있어, 제품을 선택할 때 얻을 수 있는 이익을 계산해서 가장 `이익(utility)`이 큰 제품을 선택한다고 해봅시다. 여기서 각 사람 $i$가 제품을 선택했을 때 얻는 이득을 수치적으로 나타낸 것을 $U_{i}$라고 정의했을 때, 상품을 선택하는 경우는 상품을 선택하지 않았을 때보다 이득이 가장 큰 경우입니다. 다른 말로,

$$
Y_i = 1\text{ if }U_i > 0
$$

를 의미합니다. 그렇다면 이 이익을 어떻게 계산할 수 있을까요? 경제학자들은 이러한 개인 별 유틸리티가 특정한 방정식의 형태로 나타난다고 가정을 합니다.

$$
U_{i} = \alpha + \beta X_i+\epsilon_{i}
$$

특히, 경제학자들은 모든 유틸리티를 계산할 때 모든 변수를 고려하는게 아니라, 선택하는 우리만 볼 수 있는 숨겨져있는 변수가 있다고 가정을 합니다. 어떤 `랜덤한 노이즈`가 유틸리티에 영향을 끼쳐서 선택에 영향을 주게 만들 수 있다는 것이죠. 그렇다면, $Y_i$가 1이 될 확률인 $P(Y_i = 1)$은 다음과 같이 작성할 수 있습니다. 물론 선택을 안하는 유틸리티도 어떤 랜덤 노이즈에 영향을 수 있기에 다음과 같이 쓰여집니다.

$$
P(\alpha + \beta X_i + \epsilon_{i} > \epsilon_j)
$$

랜덤 변수 $\epsilon_i$에 대한 확률 밀도 함수 (probability density function)를 $f_\epsilon(x)$, 누적 분포 함수 (cumulative distribution function)을 $F_\epsilon(x)$라고 한다면 저녀석의 확률은

$$
\int_{-\infty}^{\infty}\int_{-\infty}^{\alpha + \beta X_i + \epsilon_i}f_\epsilon(\epsilon_i)f_\epsilon(\epsilon_j)d\epsilon_j d\epsilon_i
$$

로 나타낼 수 있고, 좀 더 specific하게 나타내면

$$
\int_{-\infty}^{\infty}f_\epsilon(\epsilon_i)F_\epsilon(\alpha + \beta X_i + \epsilon_i)d\epsilon_i
$$

가 될 수 있습니다. 아무튼 우리가 $\epsilon_i$의 분포를 알거나 가정할 수 있다면 $Y_i = 1$일 확률을 계산할 수 있으며, Maximum Likelihood Estimation을 통해 적당한 $\alpha$와 $\beta$를 도출 할 수 있습니다. 그런데, 만약 $\epsilon_i$의 CDF가 다음과 같이 생겼다면 어떻게 될까요?

$$
f_\epsilon(x) = e^{-x}e^{-e^{-x}}, F_\epsilon(x) = e^{-e^{-x}}
$$

조잡하게 생긴 이 이상한 함수는 실제로 distribution이고, 놀랍게도 몇번의 계산을 통해 $\epsilon_i$가 저렇게 이상하게 생긴 생긴 distribution을 따른다면 다음과 같은 확률로 제품을 선택할 수 있다고 나옵니다.

$$
P(\alpha + \beta X_i + \epsilon_i > \epsilon_j) = \cdots = \frac{e^{\alpha + \beta X_i}}{1 + e^{\alpha + \beta X_i}}
$$

이것은 우리가 기존에 알고 있던 logistic regression과 똑같은 형태로 나오는 것을 확인할 수 있습니다. 위의 섹션에서는 그냥 임의로 $g(x)$를 정의했지만, 알고보니 유틸리티에 포함되는 random noise의 distribution의 분포와 연관되어서 나오는 것을 확인할 수 있습니다. Econometrician, 혹은 statistician들은 이러한 에러텀의 분포를 가지고 있는 녀석들을 `Gumbel Distribution`, 혹은 `Extreme Value Type 1`이라고 부릅니다.

# Extend Assumption - Probit

이제 논의를 조금 더 확장해봅시다. 저희는 logistic regression이 단순히 좋은 $g$함수를 찾아서 쓴게 아니라 통계학적 (혹은 계량경제학적)으로 **unobservable term의 distribution과 연관지어서 설명이 될 수 있다**는 것을 확인했습니다. 특히, unobservable term이 EV Type 1을 따를 때 우리가 알고 있는 logistic regression을 따른다는 것을 알게 되었습니다. 그런데, 이 error term의 distribution이 굳이 EV Type 1을 따를 필요가 없지 않을까요? 만약 정규분포를 따른다면? 아니면 이상한 다른 분포를 따른다면? 이 경우에는 어떻게 회귀식이 바뀌게 될까요?

우선, $\epsilon_i$와 $\epsilon_j$가 똑같은 분포 $G(x)$를 따른다고 가정을 합시다. 이 때 당연하게도, $\epsilon_i - \epsilon_j$역서 어떤 분포 $H(x)$를 따르겠죠. 그렇다면, 우리의 유틸리티에 따른 선택확률은 다음과 같이 기술됩니다.

$$
P(\alpha + \beta X_i + \epsilon_{i} > \epsilon_j) = H(-\alpha - \beta X_i)
$$

당연히 $H(X)$는 CDF이기 때문에 0~1사이의 값을 가지고, 이것은 valid하다는 것을 볼 수 있습니다. 이를 통해서, 똑같이 MLE를 적용하여 $\alpha$와 $\beta$를 찾을 수 있습니다. 그러나 여기서 중요한 점은, 우리가 처음에 정해야하는 분포는 $G(x)$라는 점과 동시에, $H(x)$는 웬만해서 깔끔한 형태로 나오지 않으며 웬만하면 적분을 해야하는 상태로 나타날 수 있습니다.

예를 들어서, $\epsilon_i$와 $\epsilon_j$가 `정규분포`를 따른다고 가정해봅시다. $\epsilon_i - \epsilon_j$도 정규 분포를 따르는 것이 알려져있기 때문에, $H(-\alpha - \beta X_i) = \Phi(-\alpha - \beta X_i)$입니다. 여기서 $\Phi$는 정규분포의 CDF인데, 알다시피 $\Phi$의 closed form은 존재하지 않는다는 것이 알려져있습니다. 따라서 이렇게 정규분포를 따른다고 가정하면 계산 코스트가 심하게 많이 걸려 $\beta$값을 추정하는데 오래 걸릴 수 있습니다. 그러나 이렇게 하는게 조금 더 믿음직한 가설이기 때문에 $\beta$의 설명에 있어서는 조금 더 파워풀 할 수 있습니다. 이렇게 **error term을 normal distribution으로 가정**하고 choice 확률을 회귀분석하는 방법론을 `Probit`이라 부릅니다.

당연히 정규분포 외에도 우리가 임의로 넣고 싶은 분포를 넣을 수 있습니다. 그러나 noise에 분포를 넣으면 noise의 차이가 어떤 분포를 따르는지 따로 계산을 해줘야하고, 여기서 더더욱 계산 코스트가 나갈 수 있습니다. 그러나 Logistic Regression의 경우 확률 값이 exponential 형태의 closed form으로 되어있기에 계산이 훨씬 간단하다는 이점이 있습니다. 따라서 계량 분석에서 logistic regression, 혹은 `Logit`을 많이 쓰는 이유가 그것입니다. 물론 Logit Regression과 같이 error term에 대한 가정을 EV Type 1로 둬버리면 utility framework으로 접근했을 때 말도 안되는 결론이 튀어나올 수 있습니다. 이를 해결하기 위해 Nested Logit이나 Random Coefficient Logit등의 여러가지 보완 방법론을 사용하게 되는데, 이에 대해서는 다음에 기회가 된다면 설명하도록 하겠습니다.

# 결론

이번 포스팅에서는 사람들이 머신러닝을 배울 때 Linear Regression 다음으로 배우는 분류 예측기인 Logistic Regression에 대해서 글을 작성해봤습니다. 실수 전체를 [0, 1] 사이로 연속적으로 압축시키는 Logistic Regression의 특징 외에도, utility framework에서 unobservable이 extreme value type 1을 따른다고 했을 때 상품 선택 확률이 Logistic Regression으로 나온다는 결과를 통해 logistic regression이 random noise의 분포와도 연관되어있다는 것을 확인할 수 있었습니다. 또한, random noise의 분포를 바꿔가면서 회귀 분석을 하는 방법론도 상이해지는 것을 볼 수 있었습니다.

물론 이번 포스팅도 어렵게 어렵게 다 이해할 필요 없이, 이렇게 유틸리티의 unobservable term 등 세부적인 계량경제학적인 내용을 몰라도 그냥 머신러닝을 돌려서 예측이 잘 되는 회귀분석만 선택적으로 고르면 됩니다. 그러나 통계적인 choice model을 사용한다고 했을 때, Logit과 Probit이 어떤 의미인지 정확히 모르고 사용하게 되면 추정 $\beta$값에 대해 해석하는 방법을 잃거나, 혹은 새로운 방법론이 나왔을 때 그것을 이해하는데 오랜 시간이 걸릴 것입니다. 한 번 정도는 왜 이 데이터에서는 Logit을 쓰고, 왜 이 데이터에서는 Probit을 쓰는지 고민하는 것은 양질의 데이터 사이언티스트가 될 지름길로 보입니다.
