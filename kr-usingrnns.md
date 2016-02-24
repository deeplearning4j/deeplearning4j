---
title: 
layout: kr-default
---

# DL4J의 순환 신경망 (Recurrent Neural Networks in DL4J)

이 문서는 학습 특징의 세부 사항들과 DeepLearning4J에서 그들을 사용하는 방법의 실용성을 설명합니다. 이 문서는 순환 신경망(Recurrent Neural Networks)과 그의 사용에 어느 정도 익숙함을 가정합니다 - 이는 순환 신경망을 소개하는 곳이 아니며, 그 사용과 용어 모두에 어느 정도 익숙함을 가정합니다. 만약 여러분께서 RNNs을 처음 사용하신다면, 이 페이지를 진행하기 전에 [순환 신경망 및 LSTMs로의 초보자 가이드를](http://deeplearning4j.org/lstm.html) 읽어주십시오.

**내용**


* [기본 사항: 데이터 및 네트워크 구성](#basics)
* [RNN 학습 특징](#trainingfeatures)
* [Truncated Back Propagation Through Time(시간을 통한 절단된 오류역전파](#tbptt)
* [마스킹: 일-대-다, 다-대-일 및 시퀀스 분류](#masking)
* [RNN 레이어를 다른 레이어 유형들과 결합하기](#otherlayertypes)
* [테스트 시간: 한 번에 예측 한 단계](http://deeplearning4j.org/usingrnns.html#rnntimestep)
* [시계열 데이터 가져 오기](http://deeplearning4j.org/usingrnns.html#data)
* [예제들](http://deeplearning4j.org/usingrnns.html#examples)

## <a name="basics">기본 사항: 데이터 및 네트워크 구성</a>
DL4J는 현재 순환 신경망의 한 주요 유형인 LSTM (긴 단기 메모리) 모델 (클래스 이름: GravesLSTM)을 지원합니다. 더 많은 유형들이 미래에 계획되어 있습니다.

#### RNNs를 위한 데이터

잠시 표준 피드-포워드 네트워크 (DL4J의 다층 퍼셉트론 또는 'DenseLayer')를 생각해 보십시오. 이러한 네트워크는 2차원의 입력 및 출력 데이터를 기대합니다: 즉, "형태"를 가진 데이터 [numExamples,inputSize]. 이는 피드-포워드 네트워크로의 데이터가 'numExamples' 열들/예제들을 가지고 있다는 것을 의미하고, 각각의 열은 'inputSize' 컬럼들로 구성됩니다. 하나의 예제가 형태 [1,inputSize]를 가질 것 이지만, 실제로 저희는 일반적으로 연산과 최적화된 효율성을 위해 여러 예제들을 사용합니다. 유사하게, 표준 피드-포워드 네트워크를 위한 출력 데이터 역시 2차원으로, 형태 [numExamples,outputSize]를 가질 것 입니다.

반대로, RNNs을 위한 데이터는 시계열 입니다. 따라서, 그들은 3차원을 가집니다: 시간을 위한 하나의 추가 차원. 입력 데이터는 형태[numExamples,inputSize,timeSeriesLength]를 가지고, 출력 데이터는 형태 [numExamples,outputSize,timeSeriesLength]를 가지고 있습니다. 이것은 저희의 INDArray의 데이터가 말하자면 위치 (i,j,k)에 있는 값은 minibatch에서 i번째 예제의 k번째 시간 증분에서 j번째 값으로 배치된다는 것을 의미합니다. 이 데이터 레이아웃은 다음과 같습니다.

![Data: Feed Forward vs. RNN](../img/rnn_data.png)

#### RnnOutputLayer

RnnOutputLayer는 다양한 순환 신경망 시스템과 최종 레이어로서 사용되는 (회귀 분석 및 분류 작업을 위한) 레이어의 한 유형 입니다. RnnOutputLayer는 점수 계산, 손실 함수 등을 발생시키는 오류 계산 (예측 대 실제) 같은 것들을 처리합니다. 기능적으로, 이는 ‘표준' OutputLayer 클래스 (피드-포워드 네트워크와 사용되는)와 아주 유사합니다; 그러나 그 둘 모두는 3D 시계열 데이터 세트를 출력(및 레이블/타겟으로서 예측)합니다.

RnnOutputLayer를 위한 구성은 같은 디자인 다른 레이어들을 따릅니다: 예를 들어, 분류를 위해 MultiLayerNetwork에 있는 세번째 레이어를 RnnOutputLayer로 설정합니다:

		.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")
		.weightInit(WeightInit.XAVIER).nIn(prevLayerSize).nOut(nOut).build())

실제로 RnnOutputLayer의 사용은 이 문서의 마지막에 링크된 예제들에서 보실 수 있습니다.

## <a name="trainingfeatures">RNN 학습 특징</a>

### <a name="tbptt">Truncated Back Propagation Through Time(시간을 통한 절단된 오류역전파)</a>

신경망(RNNs을 포함하여)을 학습하는 것은 연산적으로 상당한 노력을 요구할 수 있습니다. 순환 신경망의 경우, 이는 특히 저희가 긴 시퀀스를, 다시 말해 많은 시간 증분(time steps)들을 가진 학습 데이터, 처리하는 경우 입니다.

Truncated Backpropagation through time (BPTT)은 순환 신경망에서 각 파라미터 업데이트의 연산적인 복잡성을 줄이기 위해 개발 되었습니다. 요약하면, 이는 저희가 네트워크를, 연산적인 힘의 주어진 양에 비해 (보다 빈번한 파라미터 업데이트를 수행함으로써) 더 빠르게 학습하게 합니다. 여러분의 입력 시퀀스가 긴 경우 (일반적으로, 수백번의 시간 증분 이상이라면), truncated BPTT의 사용이 권장 됩니다.

길이 12 시간 증분의 시계열을 가진 순환 신경망을 학습할 때 어떻게 될지 생각해 보십시오. 여기에서 저희는 12 증분의 포워드 패스를 수행한 후, 오류를 계산하고 (예측 대 실제에 기준하여), 12 증분의 백워드 패스를 수행합니다:

![Standard Backprop Training](../img/rnn_tbptt_1.png)

위 이미지에서, 12 시간 증분을 위해서 이는 문제가 되지 않습니다. 그 대신, 입력 시계열이 10,000 혹은 그 이상 이라고 생각해 보십시오. 이 경우, 표준 backpropagation through time은 각각의 포워드와 백워드 패스 마다, 그 각각의 모든 파라미터 업데이트가 발생할 때 마다 10,000 시간 증분을 요구할 것입니다. 이것은 당연히 연산적으로 엄청난 노력을 요구합니다.

실제로, truncated BPTT는 포워드와 백워드 패스들을 더 작은 포워드/백워드 패스 작업의 세트로 분할 합니다. 이 포워드/백워드 패스 세그먼트들의 특정한 길이는 그 사용자에 의해 설정된 한 파라미터 입니다. 예를 들어, 만약 저희가 길이 4 시간 증분의 truncated BPTT를 사용한다면, 학습은 다음과 같을 것입니다:

![Truncated BPTT](../img/rnn_tbptt_2.png)

Truncated BPTT 및 표준 BPTT의 전반적인 복잡성은 대략 동일합니다 – 둘 모두 포워드/백워드 패스 동안 동일한 수의 시간 증분을 수행합니다. 그러나 이 방식을 사용하여, 저희는 대략 동일한 양의 노력으로 한 개 대신 세 개의 파라미터 업데이트를 얻습니다. 그러나 파라미터 업데이트 당 소량의 오버헤드가 있으므로 비용은 정확히 동일하지는 않습니다.

Truncated BPTT의 단점은 truncated BPTT에서 학습된 종속성들(dependencies)의 길이가 전체 BPTT보다 짧을 수 있다는 것 입니다. 이는 쉽게 확인하실 수 있습니다: 길이 4의 TBPTT로 위의 이미지들을 생각해 보십시오. 시간 증분 10에서, 그 네트워크는 정확한 예측을 하기 위해 시간 증분 0으로부터 약간의 정보를 저장할 필요가 있습니다. 표준 BPTT에서는 이것은 괜찮습니다: 그 기울기는 증분 10으로부터 증분 0까지 펼쳐진(unrolled) 네트워크를 따라 끝까지 거꾸로 흐를 수 있습니다. Truncated BPTT에서 이것은 문제가 있습니다: 시간 증분 10에서의 기울기는 요구되는 정보를 저장할 필수 파라미터 업데이트를 발생할 만큼 충분히 멀리 거꾸로 흐르지 않습니다. 이 단점은 대개 가치가 있고, (truncated BPTT 길이가 적절하게 설정되는 한) truncated BPTT는 실제로 잘 작동 합니다.

DL4J에서 truncated BPTT를 사용하는 것은 매우 간단합니다: 다음의 코드를 여러분의 네트워크 구성에 (마지막에, 여러분의 네트워크 구성에서 마지막 .build() 앞에) 추가하시기만 하면 됩니다.

		.backpropType(BackpropType.TruncatedBPTT) 
		.tBPTTForwardLength(100) 
		.tBPTTBackwardLength(100)

위의 코드 조각은 어떤 네트워크 학습이든 (즉, MultiLayerNetwork.fit() 방식으로 호출) 동등한 길이의 포워드 및 백워드 패스들을, 길이 100으로, truncated BPTT를 사용하게 할 것 입니다.

몇 가지 주의사항은:

• 기본사항으로 (만약 backprop 형식이 수동으로 지정되지 않는 경우), DL4J는 BackpropType.Standard을 사용합니다 (즉, 전체 BPTT)
•tBPTTForwardLength 및 tBPTTBackwardLength 선택 사항들이 truncated BPTT 패스들의 길이를 설정합니다. 일반적으로, 이것은 50에서 200로의 시간 증분의 순서 상 어딘가에 있습니다. 응용 프로그램에 따라 다르겠지만. 일반적으로 포워드 패스 및 백워드 패스는 같은 길이일 것 입니다 (tBPTTBackwardLength가 더 짧을 수는 있지만, 더 길지는 않습니다). 
•truncated BPTT 길이들은 전체 시계열 길이 보다 짧거나 같아야 합니다.

### <a name="masking">마스킹: 일-대-다, 다-대-일 및 시퀀스 분류</a>

DL4J는 RNNs을 위한 패딩 및 마스킹의 아이디어에 기반한 다양한 관련된 학습 특징들을 지원합니다. 패딩 및 마스킹은 저희가 일-대-다, 다-대-일을 포함한 학습 상황들을 지원하게 합니다. 또한 다양한 길이 시계열을 지원합니다 (같은 미니-배치 작업에서).

저희가 매번 시간 증분마다 발생하지는 않는 입력 혹은 출력으로 순환 신경망을 학습하기를 원한다고 가정하십시오. 이의 예제들은 (단일 예제를 위한) 아래 이미지에 표시 됩니다. DL4J는 모든 이러한 상황을 위한 학습 네트워크를 지원합니다:

![RNN Training Types](../img/rnn_masking_1.png)

마스킹 및 패딩이 없다면, 저희는 다-대-다 경우로 제한됩니다 (위, 왼쪽): 즉, (a) 모든 예제들은 같은 길이 이며, (b) 예제들은 모든 시간 증분에서 입력 및 출력 모두를 가집니다.

패딩의 배후 아이디어는 간단 합니다. 동일한 미니-배치 작업에서 길이 50 및 100 시간 증분의 두개의 시계열을 고려하십시오. 학습 데이터는 한 직사각형 어레이 입니다; 따라서, 저희는 (입력 및 출력 모두를 위한) 더 짧은 시계열을 패드 해 (즉, 0들을 추가해), 입력 및 출력이 모두 같은 길이가 되도록 합니다 (이 예제 에서는, 100 시간 증분).

물론, 이것이 저희가 한 전부라면, 그것은 학습 도중 문제를 일으킬 것입니다. 따라서, 패딩에 추가해, 마스킹 매커니즘을 사용합니다. 마스킹의 배후 아이디어는 간단합니다: 저희는 입력 또는 출력이 주어진 시간 증분 및 예제를 위해 실제로 존재하는지, 혹은 그 입력/출력이 단지 패딩을 하는지를 기록하는 두개의 추가적인 어레이를 가지고 있습니다.

RNNs로, 저희의 minibatch 데이터는 형태 [miniBatchSize,inputSize,timeSeriesLength]와 입력 및 출력을 위해 각자 [miniBatchSize,outputSize,timeSeriesLength]로, 3차원을 가진다는 것을 기억하십시오. 그 패딩 어레이는 입력 및 출력 모두를 위한 형태 [miniBatchSize,timeSeriesLength]와, 각각의 시계열과 예제를 위해 0 (‘부재’) 또는 1(‘존재’)의 값들로, 2차원 입니다. 그 입력 및 출력을 위한 마스킹 어레이들은 개별적인 어레이에 저장됩니다.

하나의 예제로, 입력 및 출력 마스킹 어레이들은 아래와 같이 보여집니다:

![RNN Training Types](../img/rnn_masking_2.png)

"Masking not required"의 경우, 저희는 마스크 어레이를 전혀 사용하지 않을 때와 동일한 결과를 제공할, 모두 1 만을 사용한 마스킹 어레이를 균등하게 사용할 수 있습니다. 또한 RNNs 학습 시 0, 1 또는 2 개의 마스크 어레이들을 사용하는 것이 가능하다는 것을 기억하십시오 - 예를 들어, 다-대-일의 경우, 출력 만을 위한 마스킹 어레이들을 가질 수 있습니다.

실제로: 이러한 패딩 어레이들은 일반적으로 데이터 import 단계에서 생성되고 (예를 들어, SequenceRecordReaderDatasetIterator에 의해 - 나중에 설명하겠습니다), DataSet 객체 내에 포함되어 있습니다. 만약 DataSet이 마스킹 어레이들을 포함하고 있다면, MultiLayerNetwork 핏(fit)은 자동으로 그것들을 학습동안 사용할 것 입니다. 그들이 부재하다면 마스킹 기능은 사용되지 않습니다.

#### 마스킹으로 평가 및 점수 매기기

마스크 어레이들은 또한 점수를 매기거나 평가를 할 때 중요합니다 (즉, RNN 분류기(classifier)의 정확성을 평가할 때). 예를 들어, 다-대-일의 경우를 고려해 보십시오: 각 예제들을 위한 오직 하나의 출력이 있고, 어떤 평가도 이 사실을 고려해야 합니다.

(출력) 마스크 어레이들을 사용하여 평가하는 것은 그것을 다음의 방식에 전달함으로써 평가하는 동안 사용될 수 있습니다:

		Evaluation.evalTimeSeries(INDArray labels, INDArray predicted, INDArray outputMask) 

labels가 그 실제 출력(3차원 시계열)인 반면, predicted는 네트워크 예측 (3차원 시계열, 레이블과 동일한 형태), 그리고 outputMask는 그 출력을 위한  2차원 마스크 어레이 입니다. 입력 마스크 어레이는 평가를 위해 필요하지 않습니다.

점수 계산 또한 MultiLayerNetwork.score(DataSet) 방식을 통해 마스크 어레이를 활용할 것 입니다. 역시, 만약 DataSet가 출력 마스킹 어레이를 포함한다면, 그것은 점수(손실 함수 – 평균 제곱 오류, 음수 로그 가능도 등)를 계산할 때 네트워크를 위해 자동으로 사용될 것 입니다.

### <a name="otherlayertypes">RNN 레이어를 다른 레이어 유형들과 결합하기</a>

DL4J에서 RNN 레이어는 다른 레이어 유형들과 결합 될 수 있습니다. 예를 들어, 동일한 네트워크에서 DenseLayer와 GravesLSTM 레이어들은 결합이 가능합니다; 또는 비디오를 위해 합성곱(Convolutional) (CNN) 레이어들과 GravesLSTM 레이어들은 결합할 수 있습니다.

물론, DenseLayer 및 합성곱 레이어들은 시계열 데이터를 처리하지 않습니다 - 그들은 다른 유형의 입력을 예측합니다. 이를 위해, 저희는 레이어 전처리기(preprocessor) 기능을 사용해야 합니다: 예를 들어, CnnToRnnPreProcessor 및 FeedForwardToRnnPreprocessor 클래스들 입니다. 모든 전처리기들을 보시려면 [여기](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/preprocessor)를 참조하십시오. 다행히 대부분의 경우, DL4J 구성 시스템이 자동으로 이 전처리기들을 필요에 따라 추가할 것 입니다. 그러나 전처리기들은 수동으로도 추가될 수 있습니다 (각 레이어마다 전처리기들의 자동 추가를 무시하면서).

예를 들어, 레이어 1과 2 사이에 전처리기를 수동으로 추가하려면, 여러분의 네트워크 구성에 다음을 추가하십시오: `.inputPreProcessor(2, new RnnToFeedForwardPreProcessor())`.
		
## <a name="rnntimestep">테스트 시간: 한 번에 예측 한 단계</a>

신경망의 다른 유형들과 마찬가지로, 예측은 `MultiLayerNetwork.output()`와 `MultiLayerNetwork.feedForward()` 방식들을 사용하여 RNNs을 위해 생성될 수 있습니다. 이 방식들은 많은 상황에서 유용할 수 있습니다; 그러나 이 방식들은 저희가 각각 그리고 매번 맨 처음으로 돌아가 시작해 시계열을 위한 예측을 생성할 수 있다는 한계를 갖습니다.

예를 들어 저희가 실시간 시스템에서 예측들을 생성하기를 원하고, 이러한 예측들은 매우 많은 양의 기록에 기반한다고 생각해 보십시오. 이 경우, output/feedForward 방식을 사용하는 것은 매번 요청될 때 마다 전체 데이터 기록에 전체 포워드 패스를 수행하기 때문에 비실용적 입니다. 만약 저희가 매 시간 증분마다 하나의 시간 증분을 위한 예측을 하고자 한다면, 이러한 방식들은 (a) 매우 비용이 많이 들고, (b) 낭비일 수 있습니다. 그들은 반복해서 같은 계산을 하기 때문입니다.

이러한 상황들을 위해, MultiLayerNetwork는 4가지 방식들을 제공합니다:

* `rnnTimeStep(INDArray)`
* `rnnClearPreviousState()`
* `rnnGetPreviousState(int layer)`
* `rnnSetPreviousState(int layer, Map<String,INDArray> state)`

rnnTimeStep() 방식은 포워드 패스(예측들)가 한번에 하나 혹은 이상의 증분들을 효율적으로 수행할 수 있도록 설계 되었습니다. 출력/피드 포워드 방식과는 달리, rnnTimeStep 방식은 요청 받을 시 RNN 레이어의 내부 상태를 추적 합니다. 저희가 이 예측들을 모두 한번에 생성하든지 (출력/피드 포워드), 혹은 이 예측들이 한번에 하나 혹은 이상의 증분들로 생성되든지 (rnnTimeStep), rnnTimeStep과 출력/피드 포워드 방식들을 위한 출력은 (각각의 시간 단계마다) 동일해야 한다는 것을 기억하는 것은 중요합니다. 따라서 유일한 차이점은 계산 비용이어야 합니다.

요약하면, MultiLayerNetwork.rnnTimeStep() 방식은 두 가지 작업을 수행합니다:

1. 이전의 저장된 상태를 사용하여 (만약 있다면), 출력/예측들(포워드 패스)을 생성하십시오.
2. 마지막 단계를 위해 활성화(activations)를 저장하여, 저장된 상태를 업데이트 하십시오 (다음에 rnnTimeStep이 호출 되었을 때 사용될 수 있도록).

예를 들어, 저희가 (입력으로서 이전의 100시간의 날씨를 기반으로) 날씨를 1시간 사전에 예측하기 위해 RNN을 사용한다고 가정하십시오. 만약 저희가 출력 방식을 사용한다면, 매 시간마다 시각 101을 위한 날씨를 예측하기 위해 100시간 전체의 데이터를 제공해야 할 것 입니다. 그런 다음 시각 102의 날씨를 예측하기 위해서 전체 100 (혹은 101) 시간의 데이터를 공급해야 할 것 입니다; 시각 103과 이상을 위해 계속.

대안으로, 저희는 rnnTimeStep 방식을 사용할 수 있습니다. 물론, 만약 저희가 첫번째 예측을 하기 전에 전체 100 시간을 사용하고자 한다면, 저희는 여전히 전체 포워드 패스를 수행해야 합니다:

![RNN Time Step](../img/rnn_timestep_1.png)

처음으로 저희가 rnnTimeStep을 요청 시 두 접근 방식들 사이의 유일한 실질적인 차이점은 마지막 단계의 활성화/상태가 저장된다는 것 입니다 - 이는 오렌지 색으로 보여집니다. 그러나 다음 번 저희가 rnnTimeStep 방식을 사용하면 이 저장된 상태는 그 다음의 예측을 위해 사용될 것 입니다:

![RNN Time Step](../img/rnn_timestep_2.png)

여기에 몇 가지 중요한 차이점들이 있습니다:

1. 두번째 이미지 (rnnTimeStep의 두번째 요청)에서 입력 데이터는 전체 데이터 기록 대신 단일 시간 증분으로 구성 됩니다.
2. 따라서 포워드 패스는 단일의 시간 증분 입니다 (수백개 - 혹은 이상과 비교해서)
3. rnnTimeStep 방식이 반환 한 후, 내부 상태가 자동으로 업데이트 됩니다. 따라서, 시각 103에 대한 예측은 시각 102와 같은 방식으로 이루어질 수 있습니다. 그리고 계속.

하지만 만약 여러분께서 새로운 (완전히 분리된) 시계열에 대한 예측을 시작하고자 한다면: `MultiLayerNetwork.rnnClearPreviousState()` 방식을 사용하여 수동으로 저장된 상태를 지우는 것이 필요 (그리고 중요) 합니다. 이것은 네트워크의 모든 순환 레이어들의 내부 상태를 초기화 할 것 입니다.

만약 여러분께서 예측에서 사용하기 위한 RNN의 내부 상태를 저장하거나 설정해야 한다면 개별적으로 각 레이어에 rnnGetPreviousState 및 rnnSetPreviousState 방식을 사용하실 수 있습니다. 이는 rnnTimeStep 방식으로부터의 내부 네트워크 상태가, 기본 설정으로 저장되지 않고 개별적으로 저장되고 로드되어야 하기 때문에, 직렬화(serialization) (네트워크 저장/로딩) 동안 예제에 유용할 수 있습니다. 이러한 get/set 상태 방식이 활성화의 유형에 의해 입력된 map을 반환하고 받아들입니다. 예를 들어, LSTM 모델에서 출력 activation과 메모리 셀 상태 모두를 저장하는 것이 필요하다는 것을 기억하십시오.

몇 가지 다른 주의 사항:

* 저희는 동시에 여러 개의 독립적인 예제/예측을 위해 rnnTimeStep 방식을 사용할 수 있습니다. 위의 날씨 예제에서, 저희는 예를 들어 같은 신경 네트워크를 사용하여 여러 위치들에 대한 예측을 하고자 할 것 입니다. 이것은 학습 및 포워드 패스 / 출력 방식들과 동일한 방법으로 작동합니다: 여러 열들이 여러 예제들에 사용됩니다 (입력 데이터에서 차원 0).
* 기록/저장된 상태가 설정되지 않으면 (즉, 초기에, 혹은 rnnClearPreviousState를 요청한 후), 기본 설정의 초기화 (zeros)가 사용됩니다. 이것은 학습 시와 동일한 접근 방식입니다.
* rnnTimeStep은 동시에 시간 증분들의 임의의 숫자를 위해 사용될 수 있습니다 - 단지 하나의 시간 단계가 아니라. 그러나, 아래를 주의하는 것이 중요합니다:
		* 하나의 시간 증분 예측의 경우: 데이터는 [numExamples,nln] 형태를 가진 2차원 입니다; 이 경우, 출력 역시 [numExamples,nOut] 형태를 가진 2차원 입니다.
		* 여러 시간 증분 예측의 경우: 데이터는 [numExamples,nln,numTimeSteps] 형태를 가진 3차원 입니다; 출력은 [numExamples,nOut,numTimeSteps] 형태를 가질 것 입니다. 역시, 최종 시간 증분 활성화는 이전과 같이 저장됩니다.

* rnnTimeStep의 요청들 사이의 예제들의 숫자를 변경하는 것은 불가능 합니다 (다시 말해, rnnTimeStep의 첫 번째 사용이 3개의 예제들을 위한 것이라면, 모든 이후의 요청들은 3개의 예제들과 함께 해야 합니다). 내부 상태를 재설정 한 후 (rnnClearPreviousState()를 사용하여), 임의의 숫자의 예제들은 rnnTimeStep의 다음 요청에 사용될 수 있습니다.
* rnnTimeStep 방식은 파라미터에 어떠한 변화도 주지 않습니다; 이는 학습 이후, 네트워크가 완성되었을 때에만 사용됩니다.
* rnnTimeStep 방식은 다른 레이어 유형들 (합성곱 또는 Dense 레이어와 같은)과 결합하는 네트워크 뿐만 아니라, 단일 및 적층된(stacked)/다중 RNN 레이어를 포함하는 네트워크와 함께 작동합니다.
* RnnOutputLayer 레이어 유형은 재발성 연결을 가지고 있지 않기 때문에 어떠한 내부 상태를 가지지 않습니다.

## <a name="data">시계열 데이터 가져 오기(importing)</a>

RNNs을 위한 데이터 가져 오기(import)는 저희가 RNNs를 위해 사용하고자 하는 데이터가 여러가지 다른 유형들을 가지고 있다는 사실로 인해 복잡해집니다: 일-대-다, 다-대-일, 가변 길이 시계열, 등. 이 섹션은 DL4J를 위한 현재까지 구현된 데이터 가져 오기 메커니즘을 설명할 것 입니다.

여기에 설명된 방식들은 SequenceRecordReaderDataSetIterator 클래스를, Canova로부터의 CSVSequenceRecordReader 클래스와 연동하여 사용합니다. 이 접근 방식은 현재 여러분께서 각 시계열이 별도의 파일에서 존재하는 장소인 파일들로부터 구분 (탭, 쉼표, 등) 데이터를 로드할 수 있게 합니다. 이 방법은 또한 다음을 지원합니다:

* 가변 길이 시계열 입력
* 일-대-다 및 다-대-일 데이터 로딩 (입력 및 레이블이 다른 파일들에 있는 경우)
* 분류를 위한 한 인덱스에서 one-hot representation으로의 레이블 변환 (즉, '2'에서 [0,0,1,0]로)
* 데이터 파일들의 시작에서 고정/지정된 번호의 열들을 건너 뛰기 (즉, 주석 또는 머릿말 열들)

모든 경우에서 데이터 파일들의 각 행은 하나의 시간 증분을 나타낸다는 점에 유의하시기 바랍니다.

(아래의 예제들에 추가하여, 여러분은 몇 가지 사용 가능한 [유닛(unit)  테스트](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetiteratorTest.java)를 찾으실 수 있습니다.)

#### 예제 1: 동일한 길이의 시계열, 별도의 파일들에서의 입력 및 레이블

여러분의 학습 데이터에 20개의 파일들에 의해 표현된 10 시계열이 있다고 가정하십시오: 각 시계열의 입력을 위한 10개의 파일, 출력/레이블을 위한 10개의 파일. 이제, 이 20개의 파일이 동일한 수의 시간 단계들을 포함한다고 가정하십시오 (즉, 동일한 수의 열들).

[SequenceRecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/SequenceRecordReaderDataSetIterator.java) 및 [CSVSequenceRecordReader](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/records/reader/impl/CSVSequenceRecordReader.java) 접근 방식을 사용하려면, 저희는 우선 두개의 CSVSequenceRecordReader 개체들, 입력을 위한 하나와 레이블을 위한 하나를 생성합니다:

		SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ","); 
		SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");

이 특정한 생성자(constructor)는 건너 뛸 열들의 숫자와 (여기에서는 1열이 생략) 분리표를 (여기에서는 쉼표가 사용됨) 사용합니다.

둘째, 저희는 어디에서 데이터를 얻는 지를 말함으로써, 이 두 리더들(readers)을 초기화 해야 합니다. 저희는 InputSplit 객체로 이를 수행합니다. 저희의 시계열이 “myInput_0.csv”, “myInput_1.csv”, …, “myLabels_0.csv”, 등의 파일 이름들로 번호 매겨져 있다고 가정하십시오. 한 가지 접근 방식은 [NumberedFileInputSplit](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/split/NumberedFileInputSplit.java)을 사용하는 것 입니다: 

		featureReader.initialize(new NumberedFileInputSplit("/path/to/data/myInput_%d.csv", 0, 9)); 
		labelReader.initialize(new NumberedFileInputSplit(/path/to/data/myLabels_%d.csv", 0, 9)); 

이 특정한 접근 방식에서, “%d”는 해당하는 숫자로 대체되고, 0에서 9까지의 숫자들이 (0과 9 모두 포함) 사용됩니다.

마지막으로, 저희는 저희의 SequenceRecordReaderdataSetIterator를 생성할 수 있습니다:

		DataSetIterator iter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression);

그 다음, 이 DataSetIterator는 그 네트워크를 학습할 MultiLayerNetwork.fit()로 전달될 수 있습니다.

miniBatchSize 인수는 각 minibatch에서 예제들 (시계열)의 개수를 지정합니다. 예를 들어, 총 10개의 파일로, 5의 miniBatchSize은 저희에게 각각 5 시계열을 가진 2개의 minibatch들을 가진 (DataSet 개체) 두개의 데이터 세트를 제공합니다.

다음을 참고 하십시오:

* 분류 문제들의 경우: numPossibleLabels은 여러분의 데이터 세트의 클래스 수 입니다. 회귀 = false 를 사용하십시오.
		* 레이블 데이터: 하나의 클래스 인덱스로서, 한 줄 당 하나의 값
		* 레이블 데이터는 one-hot representation으로 자동 변환 될 것 입니다.
* 회귀 문제들의 경우: numPossibleLabels은 사용되지 않으며 (아무 것이나 설정하십시오) 회귀 = true 를 사용하십시오.
		*입력 및 레이블에서 값들의 수는 무엇이든 될 수 있습니다 (분류와는 달리: 출력의 임의의 수를 가질 수 있습니다).
		*회귀 = true 일 때는 레이블의 어떤 처리도 완료되지 않습니다.

#### 예제 2: 동일한 파일에서 동일한 길이, 입력 및 레이블의 시계열

마지막 예제에 이어, 저희의 입력 데이터와 레이블을 위한 별도의 파일들 대신, 저희는 둘 모두를 동일한 파일에 가지고 있다고 가정하십시오. 그러나, 각각의 시계열이 여전히 별도의 파일로 존재합니다. DL4J 0.4-rc3.8로서,이 접근 방식은 출력을 위한 단일 항목이라는 제한이 있습니다 (하나의 클래스 인덱스 이거나, 혹은 단일 실제 값을 가진 회귀 출력)

이 경우, 저희는 단일 리더를 생성하고 초기화 합니다. 다시 말하지만, 저희는 하나의 머릿글 열을 건너 뛰고, 구분 쉼표와 같은 형식을 지정하고, 저희의 데이터 파일들이 `"myData_0.csv", ..., "myData_9.csv"` 라고 이름 매겨 진다고 가정합니다:

		SequenceRecordReader reader = new CSVSequenceRecordReader(1, ",");
		reader.initialize(new NumberedFileInputSplit("/path/to/data/myData_%d.csv", 0, 9));
		DataSetIterator iterClassification = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, false);

miniBatchSize 및 numPossibleLabels는 앞의 예제와 동일합니다. 여기서, labelIndex는 레이블이 어떤 항목에 있는지를 지정합니다. 예를 들어, 레이블이 다섯 번째 항목에 있는 경우, labelIndex = 4를 사용하십시오 (즉, 항목들은 0에서 numColumns-1로 색인 됩니다).

단일 출력 값에 대한 회귀 분석을 위해 저희는 이를 사용합니다:
		DataSetIterator iterRegression = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, -1, labelIndex, true);

다시 말하지만, numPossibleLabels 인수는 회귀 분석에 사용되지 않습니다.

#### 예제 3: 다른 길이의 시계열 (다-대-다)

앞의 두 가지 예제에 이어, 각 예제에 대해 개별적으로, 입력 및 레이블이 같은 길이 이지만, 이 길이들은 시계열 사이에서는 차이가 있다고 가정하십시오.

저희는, 다른 constructor는 가졌지만, 동일한 접근 방식 (CSVSequenceRecordReader 와 SequenceRecordReaderDataSetIterator)을 사용할 수 있습니다: 

		DataSetIterator variableLengthIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END); 

여기에서 인수는 AlignmentMode.ALIGN_END 추가를 제외하고는 이전의 예제와 동일합니다. 이 정렬 모드 입력은 SequenceRecordReaderDataSetIterator가 두 가지를 예상함을 알려줍니다:

1. 시계열이 다른 길이를 가질 수 있다.
2. 입력과 레이블을 정렬하기 위해서 - 각 예제 개별적으로 - 그들의 마지막 값들이 동일한 시간 단계에서 발생한다.

속성 및 레이블은 항상 동일한 길이라면 (예제 3에서의 가정과 같이), 2개의 정렬 모드 (AlignmentMode.ALIGN_END 와 AlignmentMode.ALIGN_START)는 동일한 출력을 제공할 것임을 기억하십시오. 그 정렬 모드 선택 사항은 다음 섹션에서 설명될 것 입니다.

또한, 참고 하십시오: 가변 길이 시계열은 항상 데이터 어레이에서 시각 0에서 시작합니다: 요구된다면, 시계열이 종료된 후 패딩이 추가될 것 입니다.

위의 예제 1 및 2와는 달리, 위의 variableLengthIter 경우에 의해 생성된 DataSet 개체들은 이 문서의 앞부분에서 설명된 대로 입력 및 마스킹 어레이를 포함 할 것 입니다.

#### 예제 4: 다-대-일 및 일-대-다 데이터
저희는 또한 예제 3에서 다-대-일 RNN 시퀀스 분류기를 구현하기 위해 AlignmentMode 기능을 사용할 수 있습니다. 여기에서 가정해 보겠습니다:

* 입력 및 레이블은 별도의 구분된 파일들에 있습니다.
* 레이블 파일들은 단일 열 (시간 단계)을 포함합니다 (분류를 위한 클래스 인덱스, 혹은 회귀 분석을 위한 하나 혹은 그 이상의 수)
* 입력 길이는 (선택적으로) 예제 사이에서 다를 수 있습니다.

실제로, 예제 3에서와 동일한 접근 방식은 다음을 할 수 있습니다:

		DataSetIterator variableLengthIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

정렬 모드는 비교적 간단합니다. 이들은 짧은 시계열의 시작 혹은 끝을 패드할 지의 여부를 지정합니다. 아래의 그림은 이것이 마스킹 어레이와 함께 어떻게 작동하는지 보여줍니다 (이 문서의 앞부분에서 설명한 바와 같이):

![Sequence Alignment](../img/rnn_seq_alignment.png)

일-대-다의 경우는 (위의 마지막의 경우와 유사하지만, 단지 하나의 입력으로) AlignmentMode.ALIGN_START를 사용하여 이루어집니다.

다른 길이의 시계열을 포함한 학습 데이터의 경우, 레이블과 입력은 개별적으로 각 예제들을 위해 정렬되고, 그 다음 짧은 시계열은 요구되는 대로 패드 될 것 입니다:

![Sequence Alignment](../img/rnn_seq_alignment_2.png)

#### 대안: 사용자 정의 DataSetIterator 구현하기

경우에 따라서, 여러분께서는 일반적인 데이터 가져 오기 시나리오에 맞지 않는 어떤 것을 해야할 것 입니다. 이 시나리오를 위한 하나의 선택 사항은 사용자 정의 [DataSetIterator](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.java)를 구현하는 것입니다. DataSetIterator는 단지 DataSet 개체 - 입력 및 타겟 INDArrays를 캡슐화 하고 (선택적으로) 입력 및 레이블 마스크 어레이를 추가하는 개체, 상에서 반복 처리를 하는 인터페이스 입니다.

하지만 이 접근 방식은 상당히 낮은 수준이라는 것을 참고하시기 바랍니다: DataSetIterator를 구현하는 것은 여러분께서 수동적으로 (요구 된다면) 입력 및 레이블 마스크 어레이 뿐 만이 아니라 입력 및 레이블을 위한 INDArrays를 생성하기를 요구합니다. 그러나 이 접근 방식은 여러분께 정확히 어떻게 데이터가 로드 되는지에 대한 상당한 정도의 유연성을 제공합니다.

실제로 이 접근 방식의 예제를 위해, [tex/character 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/CharacterIterator.java)와 [Word2Vec move review sentiment 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/sentiment/SentimentExampleIterator.java)를 위한 iterator를 참조하십시오.

## <a name="examples">예제들</a>

DL4J는 현재 세 가지 재발성 신경 네트워크 예제들을 가지고 있습니다:

* [캐릭터 모델링 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)로, 셰익스피어의 산문을 한 번에 하나의 문자 씩 생성합니다.
* [기본 비디오 프레임 분류 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/video/VideoClassificationExample.java)로, 비디오 (.mp4 형식)를 가져오고, 각 프레임에 존재하는 형태들을 분류 합니다.
* [Word2vec 시퀀스 분류 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/sentiment/Word2VecSentimentRNN.java)는 영화 리뷰를 긍정적인 혹은 부정적인 것들로 분류하기 위해 사전에 학습된 단어 벡터와 RNN를 사용합니다.
