---
title: Keras에서 Deeplearning4j로 모델 가져오기
layout: default
---

# Keras에서 Deeplearning4j로 모델 가져오기

*모델 가져오기는 새로운 기능입니다.  2017 년 2 월부터는 이슈를 작성하거나 버그를 신고하기 전에 master 버젼을 local에서 빌드하여 사용하거나 최신 버전을 사용하십시오.*

deeplearning4j-modelimport 모듈은 Keras를 사용하여 구성되고 훈련된 신경망 모델을 가져오기 위한 방법을 제공합니다. [Keras](https://keras.io/)는 파이썬 딥러닝 라이브러리 가운데 하나이며[Theano](http://deeplearning.net/software/theano/)나[TensorFlow](https://www.tensorflow.org) 백엔드 위에 동작하는 추상레이어(Abstraction Layer)를 제공합니다.
Keras 모델 저장은[FAQ 페이지](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) 에서 자세히 알 수 있습니다.

![Model Import Schema](../img/model-import-keras.png)

현재 Deeplearning4j에서 지원되지 않는 Keras 모델 설정을 가져오려고 할 때 바로  ‘IncompatibleKerasConfigurationException’메세지가 나타납니다. (모델 가져오기에서 지원되지 않거나  DL4J자체에서 모델, 레이어 또는 기능을 구현하지 않았기 때문임).

모델을 가져온 후에는 모델을 저장하거나 다시 로드하기 위해 자체 Modelserializer 클래스를 사용하십시오.

[DL4J gitter channel](https://gitter.im/deeplearning4j/deeplearning4j) 을 방문하면 더 많은 정보를 얻을 수 있습니다. Github을 통해 기능 요청([feature request via Github](https://github.com/deeplearning4j/deeplearning4j/issues))를 하게 되면 DL4J 개발 로드맵에 반영되게 할 수 있으며, 구현되지 않은 기능을 구현하여 반영 요청(pull request)을 할 수 있습니다.

이 페이지와 모델 가져오기 모듈은 지속적으로 업데이트될 예정이니 자주 확인하십시오!

## 인기 모델 지원

VGG16 및 기타 사전 트레이닝 된 모델들은 데모목적 및 특정 이용 사례에 대한 재훈련 용으로 널리 사용됩니다. 우리는 현재 VGG16 가져오기(or import)를 지원합니다. 뿐만 아니라 트레이닝에 적합한 데이터의 포멧과 표준화(normalization) 변환 기능, 그리고 숫자로 표현된 결과를 레이블 된 텍스트 클래스로 변환 기능을 제공하고 있습니다.

## DeepLearning4J 모델 컬렉션

DeepLearning4j는 사전 트레이닝 된 Keras 모델을 가져올 수 있을 뿐 아니라 자체 컬렉션에 적극적으로 모델을 추가합니다.

##모델 가져오기 클래스에 접근을 가능하게 하는 IDE 구성

다음 디펜던시(dependency)를 추가하여 Pom.xml을 편집하십시오.

```
<dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-modelimport</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
```

## 유용한 방법

Keras 모델 가져오기 기능을 사용하면 다음과 같은 옵션이 있습니다. Keras에는 Sequential과 Functional의 두 가지 유형의 네트워크가 있습니다. Keras Sequential 모델은 DeepLeanring4J의 MultiLayerNetwork와 같습니다. Keras의 functional 모델은 DeepLearning4J의 컴퓨테이션 그래프(Computation Graph)에 상당합니다.

## 모델 구성

현재 모든 모델이 지원되는 것은 아니나, 앞으로 무엇보다 가장 유용하고 널리 사용되는 네트워크를 가져오는 방안을 추진하고 있습니다.

이를 사용하려면 Keras에 모델을 JSON 파일로 저장해야 합니다. 사용 가능한 DeepLEarning4J 옵션은 다음과 같습니다.

* 연속모델
* 추가 트레이닝을 위한 업데이터가 있는 연속 모델
* 기능모델
*추가 트레이닝을 위한 업데이터가 있는 기능 모델

### 코드 보기

* model.to_json ()으로 Keras에 저장된 연속 모델 구성 가져오기.

```
MultiLayerNetworkConfiguration modelConfig = KerasModelImport.importKerasSequentialConfiguration("PATH TO YOUR JSON FILE)

```

* ComputationGraph 구성 가져오기, Keras에 model.to_json ()으로 저장 됨

```
ComputationGraphConfiguration computationGraphConfig = KerasModelImport.importKerasModelConfiguration("PATH TO YOUR JSON FILE)

```






## Keras에서 트레이닝 된 모델의 구성 및 저장된 웨이트(weights)

우선 Keras에서 트레이닝 된 모델의 JSON 구성과 웨이트를 모두 저장합니다. 웨이트는 H5 형식의 파일로 저장되어 있습니다. Keras에서는 웨이트와 모델 구성을 하나로 H5 파일로 저장하거나 별도의 파일로 저장할 수 있습니다.

### 코드 보기

* 연속 모델 파일

```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("PATH TO YOUR H5 FILE")

```

네트워크는 원데이터와 같은 방식으로 입력데이터를 전달하고, 포멧과 변형을 거쳐 최종적으로 정상화 하여 network.output을 호출한 뒤 추론을 위해 사용이 가능합니다.

* Sequential Model one file for config one file for weights.


```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("PATH TO YOUR JSON FILE","PATH TO YOUR H5 FILE")

```

## 추가 옵션

모델 가져오기 피처는 enforceTrainingConfig 매개 변수를 포함하고 있습니다.

오직 추측을 위해서만 사전 트레이닝 된 모델을 가져오고자 하면 enforceTrainingConfig = false로 설정해야 합니다. 지원되지 않는 트레이닝용 구성은 경보를 발생시키지만 모델 가져오기는 진행됩니다.

레이닝용 모델을 가져오거나 결과 모델이 기존 트레이닝 된 Keras 모델과 최대한 일치하는지 확인하려면 enforceTrainingConfig = true로 설정해야 합니다. 이 때 지원되지 않는 트레이닝용 구성은 UnsupportedKerasConfigurationException 를 보여주고 모델 가져오기를 중지합니다.



## Keras 모델 가져오기

아래 비디오([video tutorial](https://www.youtube.com/embed/bI1aR1Tj2DM)) 는 Keras 모델을 Deeplearning4j에 로드하고 작동중인 네트워크를 검증하는 작업 코드를 보여줍니다. Tom Hanlon강사는 Iris데이터를 통해 간단한 분류기( classifier)의 개요를 안내합니다. 이 Iris 데이터는Theano백엔드를 사용하여Keras에 내장되어 있는 데이터로Deeplearning4j로 내보낸 후 로드한 것입니다.


<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

동영상을 보는 데 문제가 있으면[view it on YouTube](https://www.youtube.com/embed/bI1aR1Tj2DM)를 클릭하십시오.

## 왜 Keras인가?

Keras는 Theano 또는 Tensorflow와 같은 Python 라이브러리 위에 있는 엡스트렉션 레이어로, 딥러닝을 위한 인터페이스를 보다 쉽게 ​​사용할 수 있는 장점을 가지고 있습니다.

Theano와 같은 프레임 워크에서 레이어를 정의하려면 우선 웨이트, 바이어스, 활성화 함수 및 입력 데이터가 출력으로 변환되는 방식을 정확하게 정의해야 합니다. 또한,  백프로파게이션(backpropagation)을 염두에 두어야 하며 그 웨이트와 바이어스도 업데이트 해야만 합니다. Keras에 이 모든 게 포함되어 있습니다. 계산 및 업데이트를 포함하는 조립식 레이어를 제공합니다.

Keras는 오직 입력의 형태, 출력의 형태 및 손실을 계산하는 방법만 정의하면 됩니다.  Keras는 모든 레이어가 올바른 크기이고 오류가 적절하게 역전파 될 수 있도록 보장합니다. 심지어 배치 작업도 합니다.

자세한 내용은[here](http://deeplearning4j.org/keras 를 참조하십시오.
