---
title: "Computation Graph를 이용해 복잡한 신경망 만들기"
layout: kr-default
redirect_from: /kr-compgraph
---

# Computation Graph를 이용해 복잡한 신경망 만들기

여기에서는 DL4J의 Computation Graph로 보다 복잡한 구조의 신경망을 만드는 방법을 소개합니다.

***주의사항: ComputationGraph는 DL4J 0.4-rc3.9혹은 상위 버전에서만 지원합니다.***

**차례**

* [Computation Graph 개요](#overview)
* [예제](#usecases)
* [ComputationGraph 신경망 구성](#config)
  * [그래프 꼭지점의 종류](#vertextypes)
  * [예제 1: Skip연결과 RNN](#rnnskip)
  * [예제 2: 다중 입력 구조와 병합 꼭지점](#multiin)
  * [예제 3: 다중 출력 구조 (Multi-Task Learning)](#multitask)
  * [전처리기 및 nIns 계산 자동화](#preprocessors)
* [ComputationGraph용 학습 데이터](#data)
  * [RecordReaderMultiDataSetIterator 예제 1: 회귀 데이터](#rrmdsi1)
  * [RecordReaderMultiDataSetIterator 예제 2: 분류 및 다중 출력 학습](#rrmdsi2)


## <a name="overview">Overview of Computation Graph</a>

DL4J의 심층 신경망에는 두 종류가 있습니다.

- [MultiLayerNetwork](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.java)는 신경망을 순차적으로 쌓은 것으로 한 층의 출력은 다음 층의 입력이 됩니다. 즉, 가장 단순하고 흔한 구조의 신경망입니다.
- [ComputationGraph](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/graph/ComputationGraph.java)는 다양하고 복잡한 신경망으로 `MultiLayerNetwork`보다 더 복잡한 구조로 신경망을 짤 수 있습니다.


다음의 경우에 ComputationGraph가 필요합니다.

- 여러개의 입력을 갖는 신경망
- 여러개의 출력을 갖는 신경망
- 층을 단순히 쌓는 것이 아니라 복잡하게 연결된 방향성 비순환 그래프(DAG; directed acyclic graph)로 연결된 구조

층을 단순히 쌓는 다는 것은 입력과 출력이 각각 하나로 되어있고 입력->층 1->층 2->출력처럼 연결하는 것입니다. 이런 경우에는 보통 `MultiLayerNetwork`를 사용합니다. `ComputationGraph`는 `MultiLayerNetwork`로 만들 수 있는 모든 구조를 다 만들 수 있지만 조금 더 설정을 해줘야하므로 단순한 신경망은 `MultiLayerNetwork`를 이용하시길 바랍니다.

## <a name="usecases">예제</a>

`ComputationGraph`를 이용하면 아래와 같은 작업을 할 수 있습니다.

- 다중 출력 학습 구조
- Skip connection을 포함한 RNN
- [GoogLeNet](http://arxiv.org/abs/1409.4842): 이미지 분류에서 쓰이는 복잡한 컨브넷 구조
- [사진 설명 자동 생성](http://arxiv.org/abs/1411.4555)
- [컨브넷을 이용한 문장 분류](http://www.people.fas.harvard.edu/~yoonkim/data/emnlp_2014.pdf)
- [ResNet](http://arxiv.org/abs/1512.03385): skip connection을 포함한 컨브넷


## <a name="config">ComputationGraph 신경망 구성</a>

### <a name="vertextypes">그래프 꼭지점의 종류</a>

ComputationGraph는 층이 아니라 여러 개의 [GraphVertex](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/graph/vertex/GraphVertex.java)를 연결하여 신경망을 구성합니다. 층([LayerVertex](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/graph/vertex/impl/LayerVertex.java) objects)도 여러가지 그래프 꼭지점 중 한 유형이며 그 외에도, 

- Input Vertices (입력)
- Element-wise operation vertices (성분별 연산)
- Merge vertices (병합)
- Subset vertices (부분집합)
- Preprocessor vertices (전처리)

등의 종류가 있습니다. 아래에서 자세히 설명합니다.

**LayerVertex**: 층 꼭지점은 신경망의 층으로 된 꼭지점으로 ```.addLayer(String,Layer,String...)``` 메쏘드로 추가합니다. 첫 번째 입력 변수는 층의 이름이고 마지막 입력 변수는 층에 이 층의 입력입니다. [입력 전처리기 (InputPreProcessor)](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor)를 직접 추가하고싶다면 ```.addLayer(String,Layer,InputPreProcessor,String...)``` 메쏘드로 추가합니다. 대부분의 경우에 이 과정은 불필요합니다 (다음 섹션에 자세히 설명합니다).

**InputVertex**: 입력 꼭지점은 ```addInputs(String...)``` 메쏘드를 사용합니. 입력 변수로 들어가는 문자열은 임의의 문자열을 사용할 수 있으며 나중에 입력 꼭지점을 참조할 때 사용합니다. 문자열의 개수는 입력의 개수를 결정하며 입력 데이터와 문자열의 순서는 일치합니다. 또, ```fit``` 메쏘드로 학습을 시작할 때 넣어주는 입력 데이터도 이 순서를 맞춰야합니다.

**ElementWiseVertex**: 성분 단위 연산 꼭지점은 성분 단위로 덧셈이나 뺄셈 연산을 수행하는데 쓰입니다. 행렬의 성분 단위로 연산이 이루어 지기 때문에 입력 데이터 등 연산과 관련된 꼭지점의 크기가 모두 같아야 합니다. 또, 이 꼭지점의 출력의 크기도 입력과 같습니다.

**MergeVertex**: 병합 꼭지점은 입력을 이어 붙이는 (concatenate) 연산을 수행합니다. 예를 들어, 각각 크기가 5와 10인 입력을 받는다면 출력의 크기는 15가 됩니다. 컨볼루션 신경망에서는 학습된 특징값 맵(feature map)을 채널 방향으로 이어 붙이는 경우가 종종 있습니다. 예를 들어 채널이 4개인 맵(4 x 너비 x 높이)과 채널이 5개인 맵(5 x 너비 x 높이)을 입력받고 이를 합쳐서 크기가 (4+5 x 너비 x 높이)인 결과를 출력합니다.

**SubsetVertex**: 부분집합 꼭지점을 이용해 꼭지점의 출력 중 일부만 사용할 수 있습니다. 예를 들어 "layer1" 꼭지점의 최초 다섯개의 값만 사용하려면 ```.addVertex("subset1", new SubsetVertex(0,4), "layer1")```를 이용합니다. 이렇게 하면 "layer1" 꼭지점의 출력 중 0번째부터 4번째까지 (0과 4 모두를 포함) 총 5개의 값만 골라냅니다.

**PreProcessorVertex**: 종종 [입력 전처리기(InputPreProcessor)](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor)에 있는 기능을 신경망 중간에 있는 층에서 사용하는 경우가 있습니다. 이 경우에 전처리기 꼭지점을 사용하면 됩니다.

마지막으로, [configuration](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/graph/GraphVertex.java)과 [implementation](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/graph/vertex/GraphVertex.java) 클래스를 이용해 필요에 맞춘 그래프 꼭지점을 만들어서 사용할 수 있습니다.


### <a name="rnnskip">Skip연결과 RNN</a>


아래 구조의 RNN을 구현해봅시다.
![RNN with Skip connections](./img/lstm_skip_connection.png)

간단한 예제이므로 입력 데이터의 크기가 5라고 가정하겠습니다. 이런 구조의 신경망은 아래와 같이 구성합니다.

```java
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        .learningRate(0.01)
        .graphBuilder()
        .addInputs("input") //can use any label for this
        .addLayer("L1", new GravesLSTM.Builder().nIn(5).nOut(5).build(), "input")
        .addLayer("L2",new RnnOutputLayer.Builder().nIn(5+5).nOut(5).build(), "input", "L1")
        .setOutputs("L2")   //We need to specify the network outputs and their order
        .build();

ComputationGraph net = new ComputationGraph(conf);
net.init();
```

`.addLayer(...)` 메쏘드에서 첫번째 입력 매개변수인 문자열 값("L1" 또는 "L2")은 해당 층의 이름이고, 마지막에 입력되는 문자열(["input"] 또는 ["input","L1"])은 해당 층의 입력입니다.


### <a name="multiin">예제 2: 다중 입력 구조와 병합 꼭지점</a>

아래와 같은 구조의 신경망을 가정해봅시다.

![Computation Graph with Merge Vertex](./img/compgraph_merge.png)

이 구조를 보면 병합 꼭지점은 층 L1과 L2의 출력을 입력으로 받습니다. 그리고 이 둘을 이어 붙입니다. 예를 들어 L1과 L2가 각각 4개의 출력을 가지고 있었따면 병합 꼭지점은 8개의 출력을 가집니다. 

이런 구조의 신경망은 아래와 같이 구성합니다.

```java
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        .learningRate(0.01)
        .graphBuilder()
        .addInputs("input1", "input2")
        .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input1")
        .addLayer("L2", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input2")
        .addVertex("merge", new MergeVertex(), "L1", "L2")
        .addLayer("out", new OutputLayer.Builder().nIn(4+4).nOut(3).build(), "merge")
        .setOutputs("out")
        .build();
```

### <a name="multitask">예제 2: 다중 출력 구조</a>

다중 출력 구조 신경망은 여러가지 작업을 동시에 수행하도록 만들어진 신경망 구조입니다. 예를 들어 두 가지 다른 분류 작업을 수행하기도 하고, 분류 작업과 회귀 작업을 동시에 수행하기도 합니다.

![Computation Graph for MultiTask Learning](./img/compgraph_multitask.png)

분류와 회귀를 동시에 수행하려면 아래와 같이 신경망을 구성합니다.

```java
ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        .learningRate(0.01)
        .graphBuilder()
        .addInputs("input")
        .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(4).build(), "input")
        .addLayer("out1", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(4).nOut(3).build(), "L1")
        .addLayer("out2", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MSE)
                .nIn(4).nOut(2).build(), "L1")
        .setOutputs("out1","out2")
        .build();
```

### <a name="preprocessors">전처리기 및 nIns 계산 자동화</a>

```ComputationGraphConfiguration```의 ```.setInputTypes(InputType...)``` 메쏘드를 이용하면 입력 데이터의 유형을 설정할 수 있습니다.

이 ```setInputTypes(InputType...)``` 메쏘드는 두 가지 역할을 합니다.

1. 우선, 이 메쏘드는 현재 상황에 필요한 [InputPreProcessor(전처리기)](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/preprocessor)를 추가합니다. 전처리기는 서로 다른 유형의 신경망의 연결을 도와줍니다. 예를 들어 컨볼루션 층과 RNN을 연결해줍니다.

2. 또 입력 데이터의 개수를 계산해줍니다. 다시 말해,  ```setInputTypes(InputType...)``` 메쏘드를 사용하면 ```.nIn(x)```을 설정해줄 필요가 없습니다. 만일 .nIn(x)를 별도로 설정하는 경우엔 그 값을 활용합니다. 즉 함수 override가 일어나지 않습니다. 

예를 들어, 신경망이 컨볼루션 입력과 feed-forward 입력을 갖는 경우에 ```.setInputTypes(InputType.convolutional(depth,width,height), InputType.feedForward(feedForwardInputSize))```을 사용하면 됩니다.

## <a name="data">ComputationGraph용 학습 데이터</a>

ComputationGraph에서 사용할 수 있는 학습 데이터는 두 가지 유형이 있습니다.

### DataSet과 DataSetIterator

DataSet과 DataSetIterator 클래스는 원래 MultiLayerNetwork에서 사용합니다. 하지만 입력과 출력이 각각 하나인 경우라면 ComputationGraph에서도 사용할 수 있습니다. 만일 다중 입/출력 구조의 신경망이라면 MultiDataset/MultiDatasetIterator를 사용합니다.

DataSet 객체는 학습 데이터의 x와 y에 해당하는 2개의 INDArrays로 구성됩니다. 또, RNN 학습을 위해 마스킹 정보를 담고있는 어레이를 추가로 포함할 수 있습니다. DataSetIterator는 DataSet객체의 iterator입니다. 



### MultiDataSet/MultiDataSetIterator

MultiDataSet은 여러 개의 입력 및 출력을 지원하는 DataSet입니다. 또, DataSet과 마찬가지로 입력/출력 개수와 같은 마스킹 어레이를 포함할 수 있습니다.

MultiDataSetIterator 사용 방법은 두 가지 입니다.

- [MultiDataSetIterator](https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/dataset/api/iterator/MultiDataSetIterator.java)를 직접 이용하는 방법이 있고,
- [RecordReaderMultiDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.java)와 DataVec record reader를 같이 사용하는 방법이 있습니다.


RecordReaderMultiDataSetIterator엔 몇 가지 옵션이 있습니다. 

- 여러 개의 DataVec RecordReader를 동시에 사용할 수 있습니다.
- 여러 개를 사용하는 경우에 다양한 유형의 RecordReader를 사용할 수 있습니다. 예를 들어 입력이 문자열인 RecordReader와 이미지인 RecordReader를 동시에 사용할 수 있습니다.
- CSV를 이용하는 경우에 여러 열(Column)을 다른 목적으로 사용할 수 있습니다. 예를 들어 0-9열을 입력 데이터로, 10-14열을 출력으로 사용할 수 있습니다.
- 정수로 된 단일 열의 인덱스를 one-hot 벡터로 변환할 수 있습니다. 


아래에 있는 RecordReaderMultiDataSetIterator 예제를 참고하십시오. [이 테스트 코드](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIteratorTest.java)도 다양한 예제를 포함하고 있습니다.

### <a name="rrmdsi1">RecordReaderMultiDataSetIterator 예제 1: 회귀 데이터</a>

이 예제는 5개의 열로 이루어진 CSV파일을 입/출력 데이터로 사용합니다. 0-2열을 입력 데이터로, 3-4열을 출력 데이터로 하는 회귀 문제의 경우 아래와 같이 MultiDataSetIterator를 설정합니다.

```java
int numLinesToSkip = 0;
String fileDelimiter = ",";
RecordReader rr = new CSVRecordReader(numLinesToSkip,fileDelimiter);
String csvPath = "/path/to/my/file.csv";
rr.initialize(new FileSplit(new File(csvPath)));

int batchSize = 4;
MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
        .addReader("myReader",rr)
        .addInput("myReader",0,2)  //Input: columns 0 to 2 inclusive
        .addOutput("myReader",3,4) //Output: columns 3 to 4 inclusive
        .build();
```


### <a name="rrmdsi2">RecordReaderMultiDataSetIterator 예제 2: 분류 및 다중 출력 학습</a>

이 예제는 입력과 출력 데이터를 각각 별도의 CSV파일에 저장해놓은 경우입니다. 그리고 출력은 두 개이며 각각 분류와 회귀에 해당합니다. 즉,

- 입력: `myInput.csv` 파일의 모든 열을 입력 값으로 사용
- 출력: `myOutput.csv` 파일을 사용하며,
  - 신경망 출력 1 - 회귀 작업, 0-3번째 열 (총 4개의 열)을 사용
  - 신경망 출력 2 - 분류 작업, 4번째 열에 각 데이터의 카테고리에 해당하는 인덱스가 저장. 즉, [0, 1, 2] 중 하나의 값을 가지며 이 값은 다시 one-hot 벡터로 변환되어야 함.

이런 경우에 아래와 같이 설정합니다.

```java
int numLinesToSkip = 0;
String fileDelimiter = ",";

RecordReader featuresReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
String featuresCsvPath = "/path/to/my/myInput.csv";
featuresReader.initialize(new FileSplit(new File(featuresCsvPath)));

RecordReader labelsReader = new CSVRecordReader(numLinesToSkip,fileDelimiter);
String labelsCsvPath = "/path/to/my/myOutput.csv";
labelsReader.initialize(new FileSplit(new File(labelsCsvPath)));

int batchSize = 4;
int numClasses = 3;
MultiDataSetIterator iterator = new RecordReaderMultiDataSetIterator.Builder(batchSize)
        .addReader("csvInput", featuresReader)
        .addReader("csvLabels", labelsReader)
        .addInput("csvInput") //Input: all columns from input reader
        .addOutput("csvLabels", 0, 3) //Output 1: columns 0 to 3 inclusive
        .addOutputOneHot("csvLabels", 4, numClasses)   //Output 2: column 4 -> convert to one-hot for classification
        .build();
```


