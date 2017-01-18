---
title: "퀵 스타트 가이드 (Quick Start Guide)"
layout: kr-default
redirect_from: /kr-quickstart
---
<!-- Begin Inspectlet Embed Code -->
<script type="text/javascript" id="inspectletjs">
window.__insp = window.__insp || [];
__insp.push(['wid', 1755897264]);
(function() {
function ldinsp(){if(typeof window.__inspld != "undefined") return; window.__inspld = 1; var insp = document.createElement('script'); insp.type = 'text/javascript'; insp.async = true; insp.id = "inspsync"; insp.src = ('https:' == document.location.protocol ? 'https' : 'http') + '://cdn.inspectlet.com/inspectlet.js'; var x = document.getElementsByTagName('script')[0]; x.parentNode.insertBefore(insp, x); };
setTimeout(ldinsp, 500); document.readyState != "complete" ? (window.attachEvent ? window.attachEvent('onload', ldinsp) : window.addEventListener('load', ldinsp, false)) : ldinsp();
})();
</script>
<!-- End Inspectlet Embed Code -->

퀵 스타트 가이드 (Quick Start Guide)
===============================

여기에서 소개하는 안내를 잘 따르면 DL4J 예제를 실행하거나 여러분의 작업을 시작할 수 있습니다.

빠른 답변이 필요한 경우엔 저희가 운영하는 [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)을 방문하십시오. 또, 채팅 페이지에서 다른 사람들의 질문/대화를 읽어보는 것 만으로도 DL4J에 대한 여러 가지를 배울 수 있을 겁니다. 만일 심층학습(딥러닝)에 대해 전혀 아는 내용이 없으시면, [시작하실때 무엇을 배워야 할지를 보여주는 로드맵](http://deeplearning4j.org/deeplearningforbeginners.html) 페이지를 참고하시기 바랍니다.

#### 맛보기 코드

Deeplearning4j는 여러 층(Layer)으로 구성된 심층 신경망(Deep neural networks)을 구성하는데 사용되는 언어입니다. 우선 `MultiLayerConfiguration`을 설정해야 합니다. 여기에서는 여러분이 사용할 신경망의 층 개수와 같은 하이퍼 파라미터(Hyperparameter)를 설정합니다. 

하이퍼 파라미터는 신경망의 구조와 학습 방법을 결정하는 매개변수입니다. 예를 들어 학습중인 모델의 계수를 몇 번 업데이트 할 지, 어떻게 계수를 초기화 할지, 어떤 활성 함수를 사용할지, 어떤 최적화 알고리듬을 사용할지 등을 결정합니다. 아래 예제 코드를 참고하십시오.

``` java
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .iterations(1)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.05)
        // ... other hyperparameters
        .backprop(true)
        .build();
```

이제 `NeuralNetConfiguration.Builder()`의 `layer()`를 호출하면 층(layer)을 추가할 수 있습니다. 이 때, 층을 추가할 위치와 입력, 출력 노드(node)의 개수, 추가할 층의 유형을 정해줘야 합니다. 예를 들면 아래와 같습니다.

``` java
        .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                .build())
```

이렇게 원하는 층을 추가한 뒤에는 `model.fit`으로 모델을 학습합니다.

## 설치를 위한 필요사항 (Prerequisites)

* [자바 (개발자 버전)](#자바) 1.7 혹은 최신 버전 (**64비트 버전만 지원**)
* [아파치 메이븐](#메이븐) (빌드 자동화 도구)
* [인텔리J IDEA](#인텔리J) 또는 이클립스
* [깃(Git)](#깃)

이 퀵 스타트 가이드를 따라하시려면 먼저 아래의 네 가지 소프트웨어를 설치해야 합니다. Deeplearning4j는 인텔리J나 메이븐같은 IDE와 빌드 자동화 도구와 배포에 익숙한 고급 자바 개발자를 대상으로 합니다. 만약 여러분이 이미 이런 소프트웨어의 사용에 익숙하시다면 DL4J를 사용하실 준비를 완벽하게 갖춘 셈 입니다.

만일 자바를 처음 시작하거나 위의 도구를 사용해본 경험이 없다면 아래에 나와 있는 설치 및 설정 안내를 따라하면 됩니다. 설치 및 사용 경험이 있다면 바로 **<a href="#examples">DL4J 예제</a>**로 넘어가면 됩니다.

#### <a name="Java">자바</a>

우선 자바 1.7 혹은 최신 버전을 [자바 개발자 도구 JDK 다운로드 페이지](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)에서 다운받으십시오. 현재 설치된 자바 버전은 아래 명령어로 확인할 수 있습니다.

 ``` shell
java -version
```

설치된 자바가 64비트인지 확인하십시오. 만일 32비트 버전의 자바가 설치되어 있다면 사용중에 `no jnind4j in java.library.path` 에러가 발생할 것입니다.

#### <a name="Maven">아파치 메이븐</a>

메이븐은 자바 프로젝트의 의존성 관리 및 빌드 자동화 도구입니다. 메이븐은 인텔리J 등 통합개발환경에 호환이 되고 이를 이용해 프로젝트 라이브러리 관리를 편하게 할 수 있습니다. [아파치에서 제공하는 메이븐 설치 가이드](https://maven.apache.org/install.html) [메이븐을 다운로드 및 설치](https://maven.apache.org/download.cgi)하면 됩니다. 현재 설치된 메이븐 버전은 아래 명령어로 확인합니다.

``` shell
mvn --version
```

맥OS에서는 패키지 관리자인 홈브류(Homebrew)를 사용합니다.

``` shell
brew install maven
```

많은 자바 개발자들이 메이븐을 사용합니다. DL4J를 원활하게 사용하려면 메이븐을 사용하기를 강력히 권장합니다. 메이븐 사용이 익숙치 않으면 [아파치에서 제공하는 메이븐 개요](http://maven.apache.org/what-is-maven.html)와 우리가 제공하는 [자바 초보 개발자를 위한 메이븐 가이드](http://deeplearning4j.org/maven.html)를 참고하십시오. 아이비나 그래들같은 [다른 빌드 도구](../buildtools)를 사용해도 되지만 스카이마인드에서는 메이븐 사용을 가정하고 있습니다.

#### <a name="IntelliJ">인텔리J IDEA</a>

[통합 개발 환경](https://ko.wikipedia.org/wiki/통합_개발_환경)을 이용하면 DL4J를 쉽게 이용할 수 있습니다. 우리가 가장 추천하는 개발 환경은 [인텔리J](https://www.jetbrains.com/idea/download/)입니다. 인텔리J와 메이븐을 함께 이용하면 의존성을 쉽게 관리할 수 있습니다. [인텔리J 커뮤니티 버전](https://www.jetbrains.com/idea/download/)은 무료로 사용 가능합니다.

이클립스나 [넷빈즈](https://ko.wikipedia.org/wiki/넷빈즈) 등 다른 개발 환경도 이용 가능하지만 가급적 인텔리J를 이용하시길 바랍니다. [DL4J Gitter 채팅방](https://gitter.im/deeplearning4j/deeplearning4j)에서 도움을 찾는 경우에도 같은 개발 환경을 사용하면 한결 수월하게 문제를 해결할 수 있습니다.

#### <a name="Git">깃(Git)</a>

[깃 최신 버전](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)을 설치하십시오. 이미 깃이 있다면 아래 명령어로 깃을 업데이트 하십시오.

``` shell
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

## <a name="examples">DL4J 예제</a>

1. 커맨드 라인에 아래 명령어를 입력하십시오.

        $ git clone https://github.com/deeplearning4j/dl4j-examples.git
        $ cd dl4j-examples/
        $ mvn clean install

2. 인텔리J를 실행하고 Import Project 메뉴를 실행합니다. 그리고 'dl4j-examples' 디렉토리를 선택하십시오. 그림은 예전에 사용하던 이름인 dl4j-0.4-examples로 되어있습니다. 

![select directory](../img/Install_IntJ_1.png)

3. 'Import project from external model'를 선택하고 메이븐이 선택되어 있는지 확인하십시오.
![import project](../img/Install_IntJ_2.png)

4. 계속 설정을 진행합니다. `jdk`로 시작하는 SDK를 선택하십시오. 그리고 완료 버튼을 누릅니다. 그러면 관련 의존 패키지 다운로드를 시작할 것입니다. 우측 하단의 진행 막대에서 진행 상황을 볼 수 있습니다.

5. 왼쪽 파일 트리에서 원하는 예제를 고르고 우클릭으로 실행하십시오.
![run IntelliJ example](../img/Install_IntJ_3.png)

## 다른 프로젝트에서 DL4J 사용하기: POM.xml 구성 방법

우선 자바 이용자는 메이븐을, 스칼라 이용자는 SBT같은 도구를 사용하기를 추천합니다. DL4J를 쓰기 위한 기본적인 의존성은 아래와 같습니다.

- `deeplearning4j-core`: 인공 신경망 구현
- `nd4j-native-platform`: DL4J의 백엔드 연산 라이브러리 ND4J의 CPU버전
- `datavec-api`: 데이터를 불러오고 벡터화하는 라이브러리

모든 메이븐 프로젝트는 POM파일을 포함합니다. 예제 파일 실행 시 [POM 파일 구성 방법(영문)](https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml)을 참고하기 바랍니다.

인텔리J를 실행하면 Deeplearning4j 예제를 골라 실행할 수 있습니다. 가장 간단한 예제를 원한다면 `MLPClassifierLinear`를 추천합니다. 예제 파일은 [여기](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java)에 있습니다.

예제를 실행하려면 우클릭 후 메뉴에서 초록색 버튼을 클릭하십시오. 그러면 하단의 창에 여러 가지 숫자가 출력됩니다. 가장 우측에 있는 숫자는 현재 사용하는 신경망의 분류 결과의 오차율입니다. 만일 학습이 잘 되고 있다면 오차율은 감소하게 됩니다. 이 창으로 실행중인 신경망 모델이 얼마나 정확하게 학습중인지 알 수 있습니다. 

![run IntelliJ example](../img/mlp_classifier_results.png)

다른 창에는 인공 신경망이 어떻게 데이터를 분류하고 있는지를 그래프로 보여줍니다. 

![run IntelliJ example](../img/mlp_classifier_viz.png)

축하합니다! 여기까지 나왔다면 Deeplearning4j로 인공 신경망을 학습시킨 것입니다. 다음 예제로 이미지 분류인 [**초보자용 MNIST**](./mnist-for-beginners)를 추천합니다.

## 다음 단계

1. 저희 Gitter 채널에 들어오십시오. 현재 3개의 채널을 운영중입니다.
  * [DL4J Live Chat](https://gitter.im/deeplearning4j/deeplearning4j): DL4J 기본 채널입니다.
  * [Tuning Help](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp): 인공 신경망 초보자를 위한 채널입니다.
  * [Early Adopters](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters): DL4J 개발과 관련된 채널입니다. 
2. [심층 신경망 소개](./neuralnet-overview) 및 [튜토리알](./tutorials) 페이지도 참고하십시오. 
3. [더 자세한 설치 안내](./gettingstarted)도 있습니다.
4. [DL4J 문서](./documentation)를 참고하십시오.

### 외부 링크

- [Maven Central의 Deeplearning4j 관련 artifacts(라이브러리)](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
- [Maven Central의 ND4J 관련 artifacts](http://search.maven.org/#search%7Cga%7C1%7Cnd4j)
- [Maven Central의 Datavec 관련 artifacts](http://search.maven.org/#search%7Cga%7C1%7Cdatavec)

### 문제 해결

**Q:** 윈도 운영체제에서 64비트 자바를 설치했는데 `no jnind4j in java.library.path` 에러가 발생합니다.

**A:** PATH에 호환되지 않는 DLL이 있을 수 있습니다. DL4J가 해당 DLL을 무시하도록 하려면 Run -> Edit Configurations -> VM Options in IntelliJ에서 아래와 같이 설정합니다.

```
-Djava.library.path=""
```
