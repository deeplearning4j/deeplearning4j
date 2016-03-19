---
title: 
layout: kr-default
---

# DL4J와 RNNs (Recurrent Neural Networks)

이 문서는 RNNs를 DL4J에서 설계/학습하는데 필요한 실용적인 내용을 다룹니다. 이 문서는 RNNs의 배경 지식을 어느 정도 갖추고 있는 독자를 대상으로 작성되었습니다. RNNs의 기본적인 내용은 [초보자를 위한 RNNs과 LSTM 가이드](http://deeplearning4j.org/lstm.html)를 참고하십시오.

**내용**

* [기본 사항: 데이터 및 네트워크 설정](#basics)
* [RNNs의 학습](trainingfeatures)
* [단기 BPTT (Back Propagation Through Time)](#tbptt)
* [마스킹: 일대다(one-to-many), 다대일(many-to-one), 및 배열 분류](#masking)
* [RNN과 다른 층의 조합](#otherlayertypes)
* [효율적인 RNNs 사용](http://deeplearning4j.org/usingrnns.html#rnntimestep)
* [시계열 데이터 가져오기](http://deeplearning4j.org/usingrnns.html#data)
* [예제](http://deeplearning4j.org/usingrnns.html#examples)

## <a name="basics">기본 사항: 데이터 및 네트워크 구성</a>
현재 DL4J는 RNNs의 여러 유형 중 LSTM(Long Short-Term Memory) 모델(클래스 이름: `GravesLSTM`)을 지원합니다. 앞으로 더 다양한 형태의 RNNs을 지원할 예정입니다.

#### RNNs과 입출력 데이터
일반적인 인공 신경망(feed-forward networks: FFNets)의 구조를 생각해봅시다 (DL4J의 `DenseLayer` 클래스). FFNets은 입력과 출력을 벡터로 표현할 수 있고, 실제 학습에서는 한 번에 여러 데이터를 읽기 때문에 2차원 데이터(데이터 개수 x 입력 벡터의 길이)를 받습니다. 이 2차원 데이터는 데이터 개수 만큼의 행과 입력 벡터의 길이와 같은 크기의 열을 갖는 행렬, 다시 말해 여러 열 벡터(column vector)의 배열(array) 입니다. 예를 들어 한번에 4개의 입력 데이터를 읽어들이고 입력 벡터가 256차원이라면 입력 데이터는 4x256 크기의 행렬입니다. 출력 데이터의 크기도 마찬가지로 계산할 수 있습니다.

RNNs은 좀 다릅니다. 기본적으로 시계열 데이터를 다루기 때문에 입력 데이터의 전체 크기는 3차원(데이터 개수 x 입력 벡터의 길이 x 전체 시간)이 되고 출력은 (데이터 개수 x 출력 벡터의 길이 x 전체 시간)이 됩니다. DL4J 문법으로 설명을 하면 `INDArray`의 `(i,j,k)` 위치의 값은 미니 배치에 있는 `i`번째 데이터에서 `k`번째 시간 단계에 있는 벡터의 `j`번째 성분입니다. 아래 그림을 참고하시기 바랍니다.


![Data: Feed Forward vs. RNN](../img/rnn_data.png)

#### RnnOutputLayer
`RnnOutputLayer`는 DL4J의 RNNs에서 출력층으로 사용하는 유형입니다. `RnnOutputLayer`는 분류/회귀 작업에 모두 사용 가능하며 현재 모델의 점수를 평가하고 오차를 계산하는 기능을 가지고 있습니다. 이런 기능은 FFNets에서 사용하는 `OutputLayer`와 비슷하지만 데이터의 모양(shape)이 3차원이라는 차이가 있습니다.

`RnnOutputLayer`를 구성하는 방법은 다른 레이어와 동일합니다. 예를 들어 아래 코드는 RNNs의 세 번째 레이어를 분류 작업을 하는 출력층으로 설정합니다.

		.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")
		.weightInit(WeightInit.XAVIER).nIn(prevLayerSize).nOut(nOut).build())

문서 하단에 실제 환경에서 이 클래스를 사용하는 예제를 링크해놓았습니다.

## <a name="trainingfeatures">RNNs의 학습</a>

### <a name="tbptt">단기 BPTT(Truncated Back Propagation Through Time)</a>
인공 신경망 학습은 연산량이 아주 많습니다. 그 중에서도 긴 배열로 RNNs을 학습하는 것은 많은 연산량을 소모합니다.

단기 BPTT는 RNNs 학습의 연산량을 줄이기 위해 개발되었습니다. 

길이가 12인 시계열 데이터로 RNNs을 학습하는 과정을 상상해보십시오. 데이터 하나로 학습을 하는데 입력->출력으로 12단계의 연산을 거치고, 출력->입력으로 다시 12단계의 backprop 연산을 합니다. (그림 참조)

![Standard Backprop Training](../img/rnn_tbptt_1.png)

12단계는 큰 문제가 아닙니다. 그러나 입력으로 들어온 시계열 데이터가 10,000개의 샘플을 가지고 있다면 어마어마한 연산량이 필요합니다. 단 한번의 계수 업데이트에 10,000단계의 입력->출력과(출력 계산) 출력->입력 과정(backprop)을 거쳐야합니다. 

이 문제를 해결하기 위해 단기 BPTT는 전체 시계열 데이터를 작게 나눠서 학습을 합니다. 예를 들어 아래 그림은 길이가 12인 시계열 데이터를 길이가 4인 작은 데이터로 나누어 학습하는 과정을 표현한 것입니다. 이 길이는 사용자가 연산량과 데이터의 크기에 따라 설정합니다.

![Truncated BPTT](../img/rnn_tbptt_2.png)

단기 BPTT와 일반적인 BPTT의 전체 연산량은 대략 비슷합니다. 그림을 보면 단기 BPTT도 결국 12번의 출력 계산과 12번의 backprop을 수행합니다. 그러나 이렇게 하면 같은 양의 데이터로 3번의 계수 업데이트가 가능합니다.

단기 BPTT의 단점은 이렇게 잘라낸 구간으로 학습할 경우 장기적인 관계를 학습하지 못한다는 점입니다. 예를 들어 위의 그림에서 t=10인 경우에 t=0일때 정보가 필요한 상황이라면 단기 BPTT는 이 관계를 학습하지 못합니다. 즉 그라디언트가 충분히 흘러가지 못하고 중간에 잘리게 되고, 결과적으로 RNNs의 '기억력'이 짧아집니다. 

DL4J에서 단기BPTT를 사용하는 방법은 아주 간단합니다. 아래의 코드를 신경망 구성(configurations)의 `.build()` 전에 입력하면 됩니다.

		.backpropType(BackpropType.TruncatedBPTT) 
		.tBPTTForwardLength(100) 
		.tBPTTBackwardLength(100)

위의 코드는 RNNs을 길이 100짜리 단기BPTT로 학습하는 코드입니다.

몇 가지 참고하실 내용이 있습니다.

* DL4J의 디폴트 설정은 단기BPTT가 아닌 일반적인 BPTT입니다.
* `tBPTTForwardLength`와 `tBPTTBackwardLength` 옵션으로 단기BPTT의 길이를 설정합니다. 보통 50-200정도의 값이 적당하고 두 값을 같은 값으로 설정합니다. (경우에 따라 `tBPTTBackwardLength`가 더 짧기도 합니다.)
* `tBPTTForwardLength`와 `tBPTTBackwardLength`은 시계열 데이터의 전체 길이보다 짧아야합니다.

### <a name="masking">마스킹: 일대다(one-to-many), 다대일(many-to-one), 및 배열 분류</a>

DL4J는 RNNs 학습과 관련한 패딩(padding) 및 마스킹(masking)을 지원합니다. 패딩과 마스킹을 이용하면 일의 아이디어에 기반한 다양한 관련된 학습 기능들을 지원합니다. 패딩 및 마스킹을 이용하면 일대다/다대일이나 가변 길이 시계열 데이터 상황에서 RNNs을 학습할 수 있습니다.

예를 들어 RNNs으로 학습하려는 데이터가 매 시간 단계마다 발생하지 않는 상황을 가정해봅시다. 아래 그림이 그런 상황입니다. DL4J를 이용하면 아래 그림의 모든 상황에 대처할 수 있습니다.

![RNN Training Types](../img/rnn_masking_1.png)

마스킹과 패딩을 쓰지 않으면 RNNS은 다대다 학습만 가능합니다. 즉, 입력 데이터의 길이가 다 같고, 출력 데이터도 입력 데이터의 길이와 같은 아주 제한된 형태만 가능합니다.

패딩(padding)의 원리는 간단합니다. 한 배치에 길이가 다른 두 개의 데이터가 있는 상황을 가정하겠습니다. 예를 들어 하나는 길이가 100이고 또 하나는 길이가 70인 경우라면, 길이가 70인 데이터에 길이가 30인 행렬을 추가해서 두 데이터가 같은 길이가 되도록 해주면 됩니다. 이 경우에 출력 데이터도 마찬가지로 패딩을 해줍니다. 

패딩을 했다면 반드시 마스킹(masking)을 해야합니다. 마스킹이란 데이터에서 어떤 값이 패딩을 한 값(그러므로 학습할 필요가 없는 값)인지를 알려주는 역할을 합니다. 즉, 두 개의 층(입력과 출력에 하나씩)을 추가해서 입력과 출력이 실제로 의미 있는 샘플인지 아니면 패딩이 된 샘플인지를 기록하면 됩니다. 

DL4J의 미니 배치에 있는 데이터는 [배치 크기, 입력 벡터 크기, 시간축 길이(timeSeriesLength)]라고 했는데, 패딩은 이 샘플이 패딩이 된건지 아닌지만 알려주면 됩니다. 따라서 마스킹 층은 [배치 크기, 시간축 길이]의 크기를 갖는 2차원 행렬입니다. 0은 데이터가 없는, 즉 패딩이 된 상태이고 1은 반대로 패딩이 아닌 실제 존재하는 데이터 샘플입니다.

아래 그림을 보고 마스킹이 어떻게 적용하는지 이해하시기 바랍니다.

![RNN Training Types](../img/rnn_masking_2.png)

마스킹이 필요하지 않은 경우엔 마스킹 층의 값을 전부 1로 설정하면 됩니다(물론 마스킹이 전혀 필요하지 않는다면 마스킹 층을 굳이 추가하지 않아도 됩니다). 또 경우에 따라 입력층이나 출력층 중 한군데에만 마스킹을 해도 됩니다. 예를 들어 다대일 학습의 경우엔 출력층에만 마스킹을 할 수도 있습니다.

DL4J 사용시 패딩 배열은 데이터를 import하는 단계에서 생성됩니다 (`SequenceRecordReaderDatasetIterator`). 그리고 나면 데이터셋 객체에 포함됩니다. 만일 데이터셋이 마스킹 배열을 포함하고 있다면 `MultiLayerNetwork` 인스턴스는 자동으로 이 마스킹 정보를 이용해 학습합니다. 

#### 마스킹을 사용한 학습 평가

학습 결과를 평가할때도 마스킹 층의 유무를 고려해야합니다. 예를 들어 다대일 분류라면 시계열 데이터를 읽고 하나를 출력하기 때문에 이 설정을 평가에 반영해야합니다.

즉, 출력 마스킹층의 값을 평가 과정에 입력해야 합니다. 아래 코드를 참고하시기 바랍니다. 

		Evaluation.evalTimeSeries(INDArray labels, INDArray predicted, INDArray outputMask) 

입력 변수는 순서대로 정답 라벨(3차원 행렬), 예측한 값(3차원 행렬), 그리고 출력 마스킹 정보(2차원 행렬) 입니다. 입력 마스킹 정보는 필요하지 않습니다. 

점수를 계산하는 `MultiLayerNetwork.score(DataSet)`는 데이터 셋을 입력으로 받는데 여기에 마스킹 정보가 포함되어 있습니다. 따라서 별도의 마스킹 정보를 입력하지 않아도 자동으로 이를 고려해 점수를 계산합니다. 

### <a name="otherlayertypes">RNNs층과 다른 층의 조합</a>

DL4J에서는 RNNs층과 다른 유형의 층을 결합하는 것이 가능합니다. 예를 들어 `GravesLSTM`과 `DenseLayer`를 연결할 수 있습니다. 비디오 데이터가 들어오는 경우엔 컨볼루션 층(Convolutional layer)과 `GravesLSTM`를 결합할 수 있습니다.

이렇게 여러 층을 결합한 신경망이 잘 작동하게 하려면 데이터를 전처리해야합니다. 예를 들어 `CnnToRnnPreProcessor`,  `FeedForwardToRnnPreprocessor`를 이용할 수 있습니다. 전처리기 목록은 [여기](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/preprocessor)에 정리되어있습니다. 대부분의 경우 DL4J는 자동으로 이 전처리기를 추가합니다. 아래의 코드를 참고하면 직접 전처리기를 추가할 수 있습니다. 이 예제는 층 1과 2 사이에 전처리기를 추가하는 코드입니다.

		.inputPreProcessor(2, new RnnToFeedForwardPreProcessor()).
		
## <a name="rnntimestep">효율적인 RNNs 사용</a>
DL4J에서 RNNs 출력은 다른 인공 신경망과 마찬가지로 `MultiLayerNetwork.output()`와 `MultiLayerNetwork.feedForward()`를 사용합니다. 주의할 점은 이 두 메서드는 늘 `시간 단계=0`에서 출발한다는 점입니다. 즉, 아무것도 없는 상태에서 새로운 시계열 데이터를 생성하는 경우에 사용하는 메서드입니다.

상황에 따라 실시간으로 데이터를 읽어오면서 결과를 출력해야 할 경우가 있습니다. 만일 그동안 누적된 데이터가 많이 있다면 이렇게 매 번 새로운 시계열 데이터를 생성하는 작업은 엄청난 연산량때문에 사실상 불가능에 가깝습니다. 매 샘플마다 전체 데이터를 다 읽어야 하기 때문입니다. 

이런 경우에는 아래의 메서드를 사용합니다.

* `rnnTimeStep(INDArray)`
* `rnnClearPreviousState()`
* `rnnGetPreviousState(int layer)`
* `rnnSetPreviousState(int layer, Map<String,INDArray> state)`

`rnnTimeStep()` 메서드는 `.output()`이나 `.feedForward()`와 달리 RNNs 층의 현재 정보를 저장합니다. 매번 과거의 데이터로 다시 연산을 수행할 필요가 없이 이미 학습된 RNNs 모델에서 `rnnTimeStep()`으로 추가된 데이터에 대한 연산만 수행하며, 그 결과는 완전히 동일합니다. 

즉, `MultiLayerNetwork.rnnTimeStep()` 메서드가 수행하는 작업은 아래와 같습니다.

1. 데이터를 입력받고, 만일 은닉층에 기존에 학습해놓은 값이 있다면 그 값을 이용해 결과를 출력합니다.
2. 그리고 기존의 학습 내용을 업데이트합니다.

예를 들어 그동안 100시간 분량의 날씨 예측을 했는데 101시간째 날씨를 예측하고 싶은 경우에, 1시간의 데이터만 추가적으로 공급하면 됩니다. 만일 이 방식이 없다면 시간이 바뀔때마다 100시간, 101시간, 102시간, 103시간.. 분량의 데이터로 다시 학습해야 합니다.




![RNN Time Step](../img/rnn_timestep_1.png)

최초에 `rnnTimeStep`이 호출되면 학습이 끝난 뒤에 은닉층의 값이 저장됩니다. 아래 그림의 우측 도식을 보면 이렇게 저장된 값을 오렌지색으로 표시했습니다. 이제 다음 입력이 들어오면 이 값을 사용할 수 있습니다. 반면 좌측은 `output()`을 사용한 경우인데, 이 경우엔 학습이 끝난 뒤에 이 값을 저장하지 않습니다.

![RNN Time Step](../img/rnn_timestep_2.png)

그 차이는 데이터가 하나 더 추가되었을 때 현격하게 벌어집니다.

1. 위 그림의 우측을 보면 단 하나의 데이터만, 즉 102시간째의 데이터만 추가된 것을 알 수 있습니다.
2. 따라서 하나의 입력만 추가적으로 학습하면 됩니다.
3. 그리고 갱신된 값은 다시 저장되기 때문에 103시간째 데이터가 추가되어도 역시 효율적인 연산을 수행할 수 있습니다.

상황에 따라 저장된 값을 지우고 완전히 새로 시작해야 할 수도 있습니다. 그런 경우엔 `MultiLayerNetwork.rnnClearPreviousState()` 메서드를 호출하면 됩니다.

만일 학습해놓은 데이터를 저장하거나 불러오길 원한다면 `rnnGetPreviousState` 및 `rnnSetPreviousState` 메서드를 이용하면 됩니다. 이 메서드는 map을 입력/반환하는데, 이 맵의 key값을 주의하시기 바랍니다. 예를 들어 LSTM 모델의 경우 출력 활성값과 메모리 셀 상태를 저장해야합니다. 

그 외 참고사항:

- `rnnTimeStep()` 메서드로 동시에 여러 개의 예측을 할 수 있습니다. 예를 들어 하나의 날씨 모델을 가지고 여러 지역의 내일 날씨를 예측하는 경우가 이에 해당합니다. 이 경우엔 각 행에 (입력 데이터의 0차원에) 각 지역의 데이터를 넣으면 됩니다.
- 만일 RNNs 모델에 기존에 저장된 정보가 없다면 (즉 최초로 실행하는 경우거나 `rnnClearPreviousState()`를 실행한 직후라면) 디폴트로 설정되어있는 초기값(0)이 사용됩니다. 
- `rnnTimeStep`은 꼭 하나의 시간 단계에만 적용될 필요가 없습니다. 예를 들어 100시간의 날씨 모델에 1시간이 아니라 여러 시간을 한번에 추가하는 것이 가능합니다. 다만 주의할 점이 있습니다.
  - 한 개의 데이터만 추가하는 경우엔 입력은 [데이터의 개수, 입력 벡터의 길이]가 됩니다. 
  - 여러 시간 단계의 데이터를 추가하는 경우엔 입출력은 3차원 행렬입니다. [데이터의 개수, 입력 벡터의 길이, 시간 단계의 개수]가 됩니다.
- 만일 처음에 `rnnTimeStep()`에 3개의 시간 단계를 사용했다면, 이후에 이 메서드를 사용할 때에도 같은 식으로 3개의 시간 단계를 사용해야합니다. 이 시간 단계를 바꾸는 방법은 `rnnClearPreviousState()`로 RNNs의 학습을 초기화하는 수 밖에 없습니다.
- `rnnTimeStep`은 RNNs모델의 전체 구조에 영향을 주지 않습니다.
- `rnnTimeStep`은 RNNs모델의 은닉층의 개수와 관계 없이 작동합니다.
- `RnnOutputLayer` 층은 피드백 연결이 없기 때문에 특별히 저장할 학습 정보를 갖고있지 않습니다. 

## <a name="data">시계열 데이터 가져오기</a>

일대다, 다대일 등 다양한 구성때문에 시계열 데이터도 다양한 종류가 필요합니다. 이제부터 DL4J에서 어떻게 데이터를 불러오는지 다루겠습니다.

여기에서는 DL4J의 `SequenceRecordReaderDataSetIterator`와 Canova의 `CSVSequenceRecordReader`를 사용하는 방법을 설명하려고 합니다. 이 방법은 시계열 데이터마다 별도의 파일로 저장된 CSV 포맷에서 데이터를 불러올 수 있습니다.
아래와 같은 경우에 사용이 가능합니다.

* 가변 길이 시계열 입력
* 일대다, 다대일 데이터 불러오기 (입력 데이터와 라벨이 별도의 파일에 저장된 경우)
* 라벨 값을 one-hot-vector로 변환 (예: [1,2] -> [[0,1,0], [0,0,1]])
* 데이터 파일에서 헤더에 해당하는 행의 데이터 건너 뛰기 (주석, 머릿말 등)

항상 데이터 파일의 각 줄(line)이 시간 단계 하나에 해당한다는 것을 유의하시기 바랍니다.

(아래의 예제와 별도로 [이 테스트 코드](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/datasets/canova/RecordReaderDataSetiteratorTest.java)를 참고하셔도 좋습니다.)

#### 예제 1: 동일한 길이의 시계열 입력/라벨이 별도의 파일에 저장된 경우

10개의 시계열 데이터로 이루어진 학습 데이터가 있다고 가정해봅시다. 즉 입력 데이터가 10개, 출력 데이터가 10개로 총 20개의 파일이 있는 경우입니다. 그리고 각 시계열 데이터는 같은 수의 시간 단계로 이루어져 있습니다. 다시 말해 행의 수가 같습니다.

[SequenceRecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/canova/SequenceRecordReaderDataSetIterator.java) 와 [CSVSequenceRecordReader](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/records/reader/impl/CSVSequenceRecordReader.java)를 사용하려면 우선 입력과 출력을 위해 두 개의 `CSVSequenceRecordReader` 인스턴스를 생성합니다.

		SequenceRecordReader featureReader = new CSVSequenceRecordReader(1, ","); 
		SequenceRecordReader labelReader = new CSVSequenceRecordReader(1, ",");

입력 변수를 보면 `1`은 데이터 파일의 맨 위 한 줄을 무시한다는 의미이고 우리가 읽어올 데이터가 콤마로 나뉘어져 있다는 것을 알려줍니다. 

이제 이 두 인스턴스를 초기화해야합니다. 여기에서 초기화는 파일의 위치를 지정해주는 과정인데, `InputSplit` 객체를 사용하겠습니다. 
파일의 이름 포맷이 `myInput_%d.csv`, `myLabels_%d.csv`라고 가정하겠습니다. [NumberedFileInputSplit](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/split/NumberedFileInputSplit.java)를 쓰면 아래와 같습니다. 

		featureReader.initialize(new NumberedFileInputSplit("/path/to/data/myInput_%d.csv", 0, 9)); 
		labelReader.initialize(new NumberedFileInputSplit(/path/to/data/myLabels_%d.csv", 0, 9)); 

이렇게 하면 0에서 9까지 (0과 9를 모두 포함) 사용합니다.

마지막으로, `SequenceRecordReaderdataSetIterator` 객체를 생성합니다.

		DataSetIterator iter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression);

이제 이 `DataSetIterator` 객체가 `MultiLayerNetwork.fit()`의 입력 변수로 전달되면 학습이 시작됩니다.

`miniBatchSize`는 미니배치의 시계열 개수를 지정합니다. 이 경우에 만일 미니 배치를 5로 지정하면 각각 5개의 시계열을 가진 미니배치 2개를 생성합니다.

아래 팁을 참고하십시오.

* 분류 문제: `numPossibleLabels`은 데이터 셋에 있는 범주의 개수입니다. `regression = false` 옵션을 지정하십시오.
  * 레이블 데이터: 한 줄에 하나의 값. (one-hot-vector가 아닌 정수)
  * 레이블 데이터는 자동으로 one-hot-vector로 변환됩니다. 
* 회귀 문제: `numPossibleLabels`의 값은 무시됩니다(아무 것이나 설정하십시오). `regression = true`로 지정하십시오.
  * 레이블 데이터: 회귀이므로 어떤 값이든지 가능합니다.
  * `regression = true`인 경우엔 라벨에 추가적인 처리(예:반올림, 범주 지정)를 하지 않습니다.

#### 예제 2: 하나의 파일에서 동일한 길이의 입/출력 시계열 데이터를 포함한 경우

이번엔 입력과 출력이 하나의 파일에 들어있는 경우를 가정하겠습니다. 이 경우에도 다른 시계열은 별도의 파일에 존재합니다. 즉 10개의 시계열이 존재하되 10개의 파일에 각각의 입력/출력을 포함하는 경우입니다. 

(DL4J 0.4-rc3.8 버전을 기준) 이 방법은 출력 하나의 열로 이루어져있어야 한다는 제한이 있습니다. 즉 [1,2,3,2,2,0,3,3,2..]같은 범주의 인덱스거나, 스칼라 값의 회귀 문제인 경우입니다.

이 경우에도 위와 비슷하지만 입/출력이 하나의 파일에 있으므로 입/출력 파일 리더를 별도로 열지 않고 하나를 사용합니다. 이번에도 파일명이 `myData_%d.csv` 포맷이라고 가정하겠습니다.

		SequenceRecordReader reader = new CSVSequenceRecordReader(1, ",");
		reader.initialize(new NumberedFileInputSplit("/path/to/data/myData_%d.csv", 0, 9));
		DataSetIterator iterClassification = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, false);

`miniBatchSize` 및 `numPossibleLabels`는 앞의 예제와 동일합니다. 추가되는 인수는 `labelIndex`인데, 이 값은 입력 데이터 행렬에서 몇 번째 열에 라벨이 있는지를 지정합니다(0을 기준으로 합니다). 예를 들어, 레이블이 다섯 번째 항목에 있는 경우, `labelIndex = 4`를 사용하십시오.

회귀 문제라면 아래 코드를 이용합니다.

		DataSetIterator iterRegression = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, -1, labelIndex, true);

`numPossibleLabels` 인수는 회귀 분석에 사용되지 않는 것을 주의하시기 바랍니다.

#### 예제 3: 다른 길이의 시계열 (다대다)

이번엔 시계열 데이터의 길이가 다양한 경우를 보겠습니다. 이 경우에도 각 데이터의 입력/출력의 길이는 같습니다. 예를 들어 2개의 데이터가 있다면 1번 데이터는 입력과 출력 모두 100의 길이를, 2번 데이터는 입력과 출력 모두 150의 길이를 갖는 경우입니다. 

이번에도 위의 예제처럼 `CSVSequenceRecordReader` 와 `SequenceRecordReaderDataSetIterator`를 사용하지만 다른 생성자를 사용합니다.

		DataSetIterator variableLengthIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END); 

인수를 잘 보면 `AlignmentMode.ALIGN_END` 추가하였고 나머지는 앞의 예제와 동일합니다. 이렇게 `AlignmentMode`를 지정해주면 `SequenceRecordReaderDataSetIterator`는 아래의 경우를 고려하여 데이터를 읽어옵니다.

1. 시계열이 다른 길이를 가질 수 있다.
2. 각 시계열의 맨 마지막 시점을 기준으로 동기화한다. 

만일 `AlignmentMode.ALIGN_START`를 사용하면 각 시계열의 맨 처음 시점을 기준으로 동기화가 일어납니다. 

또 하나 주의사항은, 가변 길이의 경우 항상 0부터 시간을 셉니다. (필요한 경우엔 뒤에 0이 패딩됩니다.)

예제 3은 마스킹 정보가 필요하기 때문에 `variableLengthIter` 인스턴스는 마스킹 배열을 포함합니다.

#### 예제 4: 다대일 및 다대다
예제 3에서 다룬 `AlignmentMode`를 이용해 RNN 다대일 분류기를 구현할 수 있습니다. 우선, 아래의 상황을 가정합니다.

* 입력 및 레이블은 별도의 파일에 저장
* 레이블은 (예제 2처럼) 하나의 열로 구성 (범주 인덱스 또는 스칼라 값 회귀 분석)
* 입력 길이는 데이터마다 달라질 수 있음

우선 아래의 생성자는 예제 3과 동일합니다.

		DataSetIterator variableLengthIter = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, miniBatchSize, numPossibleLabels, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

정렬 모드는 간단합니다. 다른 길이의 시계열을 어딜 기준으로 정렬할지를 지정합니다. 아래 그림의 좌/우를 비교하면 이해가 쉽습니다.

![Sequence Alignment](../img/rnn_seq_alignment.png)

일대다의 경우는 위의 그림에서 네 번째 경우와 비슷합니다. `AlignmentMode.ALIGN_START`를 사용하면 됩니다.

여러 학습 시계열 데이터를 불러올 경우에 각 파일 내부적으로 정렬이 이루어집니다.

![Sequence Alignment](../img/rnn_seq_alignment_2.png)

#### 다른 방법: 사용자 정의 DataSetIterator 구현하기
지금까지는 미리 구현된 클래스를 이용하는 방법을 알아봤습니다. 더 복잡한 기능이 필요한 경우엔 직접 [DataSetIterator](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.java)를 구현하는 방법이 있습니다. 간단히 말하면 `DataSetIterator`는 `DataSet` 객체를 반복 처리하는 인터페이스일 뿐 입니다.

하지만 이 방법은 상당히 로우레벨의 작업입니다. `DataSetIterator`를 구현하려면 직접 입력/레이블의 마스크 어레이를 구현하고 적합한 `INDArrays`를 생성해야합니다. 물론, 그 대신에 데이터를 정확히 어떻게 불러오고 사용하는지를 이해할 수 있고 더 다양한 학습 상황을 구현할 수 있습니다.

[tex/character 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/CharacterIterator.java)와 [Word2Vec move review sentiment 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/sentiment/SentimentExampleIterator.java)에서 사용하는 iterator를 참고하시기 바랍니다. 

## <a name="examples">예제</a>

DL4J는 현재 세 가지 RNNs 예제를 제공합니다.

* [글자(character) 모델링 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/rnn/GravesLSTMCharModellingExample.java)로, 셰익스피어의 작품을 글자(character) 기반으로 학습하고 생성합니다.
* [간단한 비디오 프레임 분류 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/video/VideoClassificationExample.java)로, 비디오 (.mp4 형식)를 불러와서 각 프레임의 객체를 분류합니다.
* [Word2vec 시퀀스 분류 예제](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/word2vec/sentiment/Word2VecSentimentRNN.java)는 영화 리뷰를 긍정적/부정적 리뷰로 분류하는 예제이며 사전에 학습된 단어 벡터와 RNNs을 사용합니다.
