---
title: "분류 모델의 평가 지표 알아보기"
categories:
  - data-analysis
tags:
  - 일반 데이터 분석론
  - 머신러닝
toc: true
sidebar:
  nav: data
use_math: true
---

# 들어가기 앞서...

이 내용을 이해하기 위해서는 중학교 수준의 수학 능력 가지면 됩니다!

# 분류 문제를 푸는 원리

대부분의 일반 사람들이 머신러닝 기법 등의 예측 AI를 통해 가지고 싶은 것은 구별 능력입니다. 이 데이터를 가지고 있는 사람들이 이번 달에 구매할까? 이런 RGB 정보를 가지고 있는 사진이 고양이일까? 이러한 구별을 하기 위해서 머신러닝을 사용하여 분류기, `classifier`를 만들려고 합니다. 그러나, 사실은 머신러닝은 데이터를 구별해주지 않습니다. 각 데이터별로 어떤 분류일지 "확률”을 추정해줍니다. 특히, 우리가 예측하고자 하는 값이 어떤 특성으로 분류되는지가 아니라, 이 데이터가 어떤 클래스에 들어갈지 말지에 대한 binary 변수라면, classifier는 클래스에 들어갈 **확률을 추정해줍니다**. 그렇다면 이 상황에서 어느 정도 확률이 높으면 1이고, 그게 아니면 0이라고 정의할 수 있을텐데, 어떤 threshold를 기준으로 추정을 해야할지 감이 안올 수 있습니다. 수량을 예측하는 regression의 경우에는 간단하게 MSE(Mean Squared Error)나 MAE(Mean Absolute Error)를 통해 모델의 예측력을 평가할 수 있지만, binary classification에서는 threshold를 2차적으로 계산을 해야하기 때문에 예측력을 측정하기 다소 애매합니다. 그렇다면, 이 threshold를 어떻게 정의하면 좋을까요?

만약 우리가 threshold $(th)$ 를 임의로 정했다고 합시다. 그러면 우리의 예측치 $P_i > th$이면 1, $P_i \leq th$이면 0으로 정의를 해봅시다. 즉,

$$
\hat{Y}_i=\begin{cases}1 &\text{if }P_i > th\\ 0 &\text{if }P_i \leq th\end{cases}
$$

인 상황입니다. 그러면 당연히 $\hat{Y}_i \in \{0, 1\}$입니다. 이 상황에서 우리는 true Y값인 $Y_i$와 predicted Y값인 $\hat{Y}_i$가 존재합니다. 이렇게 이진 분류에 대한 예측치와 실제 값이 있을 때, 분류 예측력을 평가할 수 있는 좋은 메트릭 $Metric(Y, \hat{Y}; th)$가 있다고 했을 때, threshold는 다음과 같이 세팅할 수 있습니다.

$$
th = \argmax_k Metric(Y, \hat{Y}; k)
$$

여기서 어떤 분류 metric을 사용하면 좋을까요? 이는 상황, 데이터마다 다르겠지만, 일반적으로 쓰이는 것들을 한 번 의미를 짚어가며 보도록 하겠습니다. 이번 포스팅에서 저희가 볼 것은 다음 4가지입니다.

1. Accuracy
2. Precision
3. Recall
4. F1 Score

네가지 metric 모두 $\hat{Y}_i$와 $Y_i$를 인풋으로 해서 나오는 0과 1 사이의 값이며, 1로 가면 갈 수록 예측력이 좋다고 보통 이야기 할 수 있습니다. 그렇다면 각각에 대해서 살펴보도록 하겠습니다.

# Accuracy - 가장 intuitive한 metric

우리가 생각했을 때, 예측을 가장 잘했다는 것은 바로 $\hat{Y}_i = Y_i$인 $i$의 값들이 많다는 것입니다. 다른 말로 하면, 예측치와 실제치가 같은 데이터의 비율이 많을 수록 예측을 잘했다고 할 수 있는 것이죠. 이 정신으로 나온 metric이 accuracy입니다. 수식적으로 표현하자면,

$$
\operatorname{Acc}(Y, \hat{Y}) = \frac{|Y_i = \hat{Y}_i|}{|Y_i|}
$$

가 됩니다. 이런 accuracy가 항상 높으면 좋겠지만, 실제로는 이렇게 accuracy가 항상 1로 가지는 않습니다. 대부분 데이터 분석 및 예측 모델을 만들었을 때 accuracy는 0.9 혹은 그 아래일 것입니다. 그러면 이 때 다음과 같은 고민을 자연스럽게 할 수 있습니다. “accuracy 올리는건 포기!, 같은 accuracy라 하더라도 어떤 모델이 더 성능이 좋다고 말할 수 있을까?” 이 문제를 해결하기 위해 등장한 것인 Recall과 Precision입니다.

# Recall & Precision - False Positive, True Negative

Accuracy 메트릭이 만들어진 기저는 “예측치와 실제치가 같은지 여부에 대해 관심”입니다. 그러나, 사실 예측치와 실제치를 잘 뜯어보면 다음과 같은 4가지의 경우가 존재함을 확인할 수 있습니다.

... 아 갑자기 쓰기 귀찮다

# F1 Score - Combination of Recall and Precision

Recall과 Precision을 동시에 고려하기 위해 F1 Score라는 것을 개발

F1 Score = Recall과 Precision의 조화평균

조화평균? 역수 평균의 역수 ⇒ 왜 조화평균 사용? 분모에 수정을 가하려고

원래 분포 TP + FN or TP + FP ⇒ TP + (FN과 FP의 평균)로 둬서 하면 Recall과 Precision을 둘 다 포용할 수 있을 것 같음 ㅋㅋ

# 결론

Acc, Recall, Precision, F1 Score 이런 애들은 사실 다 그냥 discretization을 해버리지만, probability 자체를 numeric하게 풀어버리는 metric이 있음

- Log-loss
- ROC Curve & AUC

이런 애들은 probability 자체를 사용해서 모델의 성능을 계산할 수 있음. 그러나 이렇게 갈수록 더 비직관적인 메트릭이 되어버리기 때문에 초보자들은 Acc, Rec, Prec, F1까지 알아도 충분할 것 같음
