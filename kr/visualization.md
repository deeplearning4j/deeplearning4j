---
title: 신경망 학습에서 시각화, 모니터링, 디버그 하는 방법
layout: kr-default
redirect_from: kr/kr-visualization
---

# 네트워크 학습의 시각화, 모니터링 및 디버그

내용

* [Deeplearning4j 트레이닝 UI를 통한 네트워크 교육 시각화](#ui)
    * [Deeplearning4j UI: 개요 페이지](#overviewpage)
    * [Deeplearning4j UI: 모델 페이지](#modelpage)
* [Deeplearning4J UI 및 스파트 트레이닝](#sparkui)
* [UI 사용을 통한 네트워크 튜닝](#usingui)
* [TSNE 및 Word2Vec](#tsne)

## <a name="ui">Deeplearning4j 트레이닝 UI를 통한 네트워크 교육 시각화</a>

**참고**: 이 정보는 DL4J 0.7.0 버전 및 상위버전에 적용됩니다.

DL4J는 당신의 브라우저에서 실시간으로 현재 네트워크 상태 및 트레이닝의 진행상태를 시각화할 수 있는 사용자 인터페이스를 제공합니다. UI는 일반적으로 신경망을 조정하는 데 사용됩니다. 예를 들어, 좋은 결과를 얻기 위해 하이퍼 파라미터 (학습 속도와 같은)를 선택하는 데 도움을 줍니다.


**1단계: Deeplearning4j UI 디펜던시를 프로젝트에 추가하기**

```
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-ui_2.10</artifactId>
        <version>${dl4j.version}</version>
    </dependency>
```

접미사 ```_2.10``` 에 유의: 이것은 스칼라버전입니다. (백엔드로 스칼라 라이브러리인 Play프레임워크를 사용하기 때문에) 다른 스칼라 라이브러리를 사용하지 않는 경우 ```_2.10```이나 ```_2.11```모두 괜찮습니다.


**2단계: 프로젝트에서 UI사용하기**

비교적 간단합니다:

```
    //Initialize the user interface backend
    UIServer uiServer = UIServer.getInstance();

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage);

    //Then add the StatsListener to collect this information from the network, as it trains
    net.setListeners(new StatsListener(statsStorage));
```

UI에 접근하려면 ```http://localhost:9000/train```로 이동하십시오. ```org.deeplearning4j.ui.port```의 시스템 속성을 사용하여 포트를 세팅할 수 있습니다. 즉, 포트 9001을 사용하려면 시작할 때 JVM에 다음을 전달하십시오. ```Dorg.deeplearning4j.ui.port=9001```

그 후 네트워크에 ```fit```한 방법을 호출하면 정보가 수집되고 UI로 라우팅 될 것입니다.


**예시:** [UI 예시를 참고하십시오.](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface/UIExample.java)

UI예제 전체세트는 [여기](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface)에서 확인하실 수 있습니다.


### <a name="overviewpage">Deeplearning4j UI: 개요 페이지</a>

![Overview Page](./img/DL4J_UI_01.png)

이 개요 페이지(3페이지 중 하나에 해당)는 아래와 같은 정보를 포함하고 있습니다:

- 좌측 상단: 스코어 vs 이터레이션 차트 – 현 미니배치의 손실함수 값
- 우측 상단: 모델 및 트레이닝 정보
- 좌측 하단: 파라미터 비율 대 모든 네트워크의 웨이트와 이터레이션에 대한 업데이트
- 우측 하단: 표준편차 (vs 시간): 활성화, 변화도 및 업데이트

하단의 두 차트는, 값의 대수(10진수)로 표시됩니다. 그래서 업데이트의 파라미터의 비율 차트에서의 -3은 10<sup>-3=0.001</sup>에 해당합니다.

업데이트에 대한 파라미터의 비율은 정확히 말해 이 값들의 평균 크기의 비율입니다.

뒷부분에서 이러한 값을 실제로 사용하는 방법을 확인할 수 있습니다.


### <a name="modelpage">Deeplearning4j UI: 모델 페이지</a>

![Model Page](./img/DL4J_UI_02.png)

모델페이지는 선택 메커니즘으로 작동하는 신경망 레이어의 그래프를 포함하고 있습니다. 레이어를 클릭하면 정보를 볼 수 있습니다.

오른쪽에서 레이어를 선택하면, 다음의 차트를 볼 수 있습니다.

-	레이어 정보 테이블
-	개요 페이지에 따라 이 레이어의 파라미터 비율로 업데이트 합니다. 이 비율(파라미터 및 업데이트 평균 크기)의 구성 요소는 탭을 통해서 사용이 가능합니다.
-	시간 경과에 따른 레이어 활성화 (평균, 평균 +/- 2 표준편차)
-	파라미터의 유형에 따른 파라미터와 업데이트의 막대그래프
-	학습율 vs 시간 (학습율 일정이 사용되지 않았다면 일정함(flat))


*참고: 파라미터는 웨이트(W)와 편향(b)으로 표시됩니다. 회귀신경망에서 W는 레이어와 아래 레이어를 연결하는 웨이트를 말하며, RW는 회귀 웨이트를 말합니다. (시간 간격 사이의 웨이트)*


## <a name="sparkui">Deeplearning4J UI 및 스파크 트레이닝</a>

DL4J UI는 스파크와 함께 사용가능합니다. 하지만 0.7.0버전에서 의존성(dependencies)이 충돌하는 것은 UI와 스파크의 실행이 같은JVM이지만 어렵다는 것을 의미합니다.

가능한 두가지 대안:

1.	관련된 통계를 수집하여 추후에 오프라인에서 시각화합니다.
2.	별도의 서버에서 UI를 실행하고, 원격UI 기능을 사용하여 스파크 마스터에서 UI인스턴스로 데이터를 업로드합니다.


**추후 오프라인에서의 사용을 위한 통계수집**

```
    SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

    StatsStorage ss = new FileStatsStorage(new File("myNetworkTrainingStats.dl4j"));
    sparkNet.setListeners(ss, Collections.singletonList(new StatsListener(null)));
```

다음을 사용하여 나중에 저장된 정보를 로드하고 표시할 수 있습니다:

```
    StatsStorage statsStorage = new FileStatsStorage(statsFile);    //If file already exists: load the data from it
    UIServer uiServer = UIServer.getInstance();
    uiServer.attach(statsStorage);
```


**원격 UI기능 사용하기**

UI를 실행하는 JVM에서:

```
    UIServer uiServer = UIServer.getInstance();
    uiServer.enableRemoteListener();        //Necessary: remote support is not enabled by default
```
이를 위해 ```deeplearnin4j-ui_2.10```이나 ```deeplearning4j-ui_2.11``` 디펜던시가 필요합니다.

스파크 트레이닝 인스턴스에서:

```
    SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

    StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://UI_MACHINE_IP:9000");
    sparkNet.setListeners(remoteUIRouter, Collections.singletonList(new StatsListener(null)));
```
스파크에서 디펜던시 충돌을 피하기 위해서는 ```deeplearning4j-ui_2.10``` UI디펜던시 *전체가 아닌* ```deeplearning4j-ui-model```을 사용하여 StatsListener를 얻어야 합니다.


주의: ```UI_MACHINE_IP```를 사용하는 컴퓨터의 유저 인터페이스 인스턴스IP주소로 바꾸십시오.


## <a name="usingui">UI 사용을 통한 네트워크 튜닝</a>

신경망 학습의 시각화에 자세한 자료는 [web page by Andrej Karpathy](http://cs231n.github.io/neural-networks-3/#baby)를 보십시오. 이 웹페이지의 내용을 먼저 이해하는 것이 좋습니다.

신경망을 조정하는 것은 때때로 과학이라기보다는 예술쪽에 가깝습니다. 아래의 유용한 아이디어를 소개드립니다.


**개요 페이지 – 모델 스코어 vs. 이터레이션 차트**

스코어 vs. 이터레이션은 전반적으로 시간의 경과에 따라 내려갑니다.

- 스코어가 안정될때까지 학습율을 줄여보십시오.
- 증가한 스코어는 잘못된 데이터 정규화와 같은 다른 네트워크 문제를 나타내는 것일 수도 있습니다.
- 만일 스코어에 변함이 없거나(flat) 매우 천천히 줄어든다면 (수백번의 이터레이션에 걸쳐) (a) 학습율이 너무 낮게 설정되었거나 (b) 최적화에 어려움을 겪을 수 있습니다. 후자의 경우 SDG 업데이터를 사용하고 있다면, Nesterovs (momentum), RMSProp, Adagrad와 같은다른 업데이터를 사용해보십시오.
- 섞이지 않은 데이터는 매우 대략적이며 비정상적인 스코어 vs 이터레이션 그래프를 만들어낼 수도 있습니다. (예를 들어 각 미니배치가 분류를 위해 단 한개의 클래스만 포함하고 있는 경우)
- 이 선(line)차트에서는 일부 잡음이 발생이 예상됩니다. (선이 위아래로 작은범위에서 움직임) 그러나, 런스 변화(runs variation) 사이에서 스코어가 상당히 다양하거나 매우 크면 이것은 문제가 될 수 있습니다.
    - 위에 언급한 문제 (학습율, 정규화, 데이터 셔플) 가 문제의 원인이 될 수 있습니다.
    - 미니배치의 크기를 아주 작은 예제의 수로만 설정한다면, 스코어 vs. 이터레이션 그래프에 잡음을 발생시킬 수 있으며, 최적화도 어렵게 만들 *가능성* 이 있습니다.


**개요 페이지 및 모델 페이지 - 업데이트 사용한 파라미터 비율 차트**

- 개요 페이지나 모델 페이지에서 업데이트에 대한 파라미터의 평균크기의 비율을 확인할 수 있습니다.
    - “평균 크기” = 파라미터 혹은 현재 시간 단계에서 업데이트의 절대값 평균
- 이 비율의 가장 중요한 용도는 학습율을 선택하는 데에 있습니다. 우선적으로 이 비율은 약 1:1000=0.001이어야 합니다. Log10 차트에서는 이것이 -3에 해당하는 값입니다. (즉, 10-3=0.001)
    - 이것은 대략적인 설명이며 모든 네트워크에 다 적절하지는 않을 것입니다. 하지만 좋은 시작점이 될 수 있습니다.
    - 비율이 여기에서 크게 벗어나면 (예 :> -2 (즉, 10-2 = 0.01) 또는 <-4 (즉, 10-4 = 0.0001)) 파라미터가 유용한 기능(features)을 배우기에는 다소 불안정해지거나 너무 느리게 변할 수 있습니다.
    - 이 비율을 조정하려면, 학습율을 조정하십시오. (혹은 파라미터 초기화) 일부 네트워크에서에서는 각각의 레이어에 다른 학습율을 적용해야 할 수도 있습니다.
- 비율에서 비정상적으로 큰 스파이크가 발생하는지 지켜보십시오. 폭발적인 경사를 나타낼 수 있습니다.


**모델 페이지: 레이어 활성화 (vs. 시간) 차트**

이 차트는 레이어가 사라진다거나 폭발적인 활성화를 감지하는 데에 사용할 수 있습니다. (부실한 웨이트 초기화, 너무 많은 정규화, 부족한 데이터 정규화, 혹은 지나치게 높은 학습율)

- 이 차트는 시간이 지나면서 이상적으로 안정됩니다. (보통 수백회 정도의 이터레이션)
- 활성화에 대한 표준편차는 약 0.5~2.0입니다. 이 범위를 상당히 벗어났다면 위에 언급한 문제 중 하나가 발생한 것일 수 있습니다.


**모델 페이지: 레이어 파라미터 막대그래프**

레이어 파라미터 막대그래프는 가장 최신의 이터레이션만을 표시합니다.

- 웨이트의 경우 막대그래프는 시간이 지나면서 약 Gaussian (정규) 분포를 나타내야 합니다.
- 바이어스의 경우 막대그래프는 보통 0에서 시작되며 Gaussian 분포와 비슷한 모양으로 끝납니다.
    - 한가지 예외가 있는데, 바로 LSTM 회귀신경망입니다. 하나의 게이트(the forget gate)의 바이어스는 학습 디팬던시를 장시간에 걸쳐 돕기 위해 구성도 가능 하지만 기본 1.0으로 설정됩니다. 결과적으로 바이어스 그래프는 처음에 약0.0정도가 되며 또다른 바이어스 세트는 1.0 정도가 됩니다.
- +/-로 무한 갈라지는 파라미터에 주의하십시오. 학습율이 너무 높거나 충분하지 않은 정규화 때문일 수 있습니다. (네트워크에 L2 정규화를 추가합니다)
- 큰 바이어스에도 주의하십시오. 클래스의 분포가 불균형하다면, 분류를 위한 아웃풋 레이어에서 가끔 발생할 수 있습니다.


**모델 페이지: 레이어 업데이트 막대그래프**

레이어 파라미터 막대그래프는 가장 최신의 이터레이션만을 표시합니다.

- 업데이트 – 학습율, 모멘텀, 정규화 등을 적용한 *후* 의 경사도
- 파라미터 그래프와 마찬가지로, Gaussian분포를 보여야 합니다.
- 큰 값에 주의하십시오. 네트워크에서 폭발적인 경사도를 나타낼 수 있습니다.
    - 폭발하는 경사도는 네트워크 파라미터를 엉망으로 만들 수 있습니다.
    - 웨이트 초기화, 학습율 입력/레이블 데이터 정규화 문제를 발생시킬 수 있습니다.
    - 회귀 신경망의 경우, [경사도 정상화 혹은 경사도 클리핑](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/conf/GradientNormalization.java)을 추가하면 도움이 될 수 있습니다.


**모델 페이지: 파라미터 학습율 차트**

이 차트는 선택된 파라미터의 학습율을 보여줍니다.

만일 학습율 스케쥴을 사용하지 않는다면, 차트는 변함이 없을 것입니다. (flat) 하지만, 학습율 스케쥴을 사용하면, 차트를 통해 시간 경과에 따른 학습율의 현재값(각 파라미터에 대한)을 추적할 수 있습니다.


## <a name="tsne">TSNE 및 Word2vec</a>

우리는 [TSNE](https://lvdmaaten.github.io/tsne/)를 사용하여 [단어 특징(feature) 백터](./word2vec.html)와 프로젝트 단어의 차원을 2 차원이나 3 차원 공간으로 축소합니다. 다음은 Word2Vec에서 TSNE를 사용하기 위한 몇 가지 코드입니다:

        log.info("Plot TSNE....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);
