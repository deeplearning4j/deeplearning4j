---
title: 
layout: kr-default
---

# Deeplearning4j란 무엇인가요?

Deeplearning4j는 자바(Java)와 스칼라(Scala)를 위해 작성된 최초의 상용 수준 오픈소스 딥러닝 라이브러리입니다. 연구 목적으로 쓰여진 다른 라이브러리와 달리 Deeplearning4j는 상용 서비스를 위해 하둡(Hadoop)/스파크([Spark](../gpu_aws.html))와 통합해 사용할 수 있도록 설계되었습니다. 스카이마인드([Skymind](http://skymind.io))는 Deeplearning4j의 유료 사용자를 지원하는 회사입니다.

Deeplearning4j는 최신 기술을 간편하게 사용하는데 초점을 두었습니다. 특히, 설치와 활용에 있어서 일반적인 문법과 규칙을 사용하였기 때문에 머신 러닝에 대한 깊은 지식이 없는 사람도 빠르게 시제품을 만들 수 있도록 하고 있습니다. 확장성에도 초점을 두어 어떠한 규모의 데이타에도 사용할 수 있습니다. Deeplearning4j는 아파치 2.0 라이센스로 배포되기 때문에 파생된 모든 소스 코드는 저작권이 코드의 작성자에게 귀속됩니다.

지금 바로 Deeplearning4j 예제 코드를 실행해보세요. [빠른 설치 페이지의 안내](http://deeplearning4j.org/quickstart.html)를 따르면 인공 신경망 예제를 실행할 수 있습니다.

### [딥 러닝 활용 사례](http://deeplearning4j.org/use_cases.html)

* [얼굴/이미지 인식](../facial-reconstruction-tutorial.html)
* 음성 검색
* 음성 인식 및 음성-문자 변환
* 스팸 메일 필터링 (비정상 행위 탐지)
* 전자 상거래 사기 탐지
* 추천 시스템 (고객관리, 고객 유지, 광고 기술)
* [회귀 분석](http://deeplearning4j.org/linear-regression.html)

### DL4J의 강점

* 다목적 N차원 배열 클래스([n-dimensional array class](http://nd4j.org/))
* [GPU](http://nd4j.org/gpu_native_backends.html) 통합
* [하둡](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn) 및 [스파크](http://deeplearning4j.org/gpu_aws.html)로 쉽게 [확장 가능](http://deeplearning4j.org/spark.html)
* [카노바(Canova)](http://deeplearning4j.org/canova.html): 기계 학습을 위한 벡터 처리 기술
* [ND4J: Numpy의 두 배 속도를 자랑하는 선형 대수 라이브러리](http://nd4j.org/benchmarking)

Deeplearning4j는 분산 처리와 단일 스레드 처리를 모두 포함합니다. 분산 처리의 경우 동시에 여러 서버 클러스터에서 학습을 할 수 있으며 결과적으로 대량의 데이터를 신속하게 처리할 수 있습니다. //from here
망들(nets)은 [반복 감소](http://deeplearning4j.org/iterativereduce.html)를 통해 병렬로 학습되며, 자바, [스칼라](http://nd4j.org/scala.html) 및 [Clojure](https://github.com/wildermuthn/d4lj-iris-example-clj/blob/master/src/dl4j_clj_example/core.clj)와 균일하게 호환 가능합니다. 오픈 스택의 모듈식 구성 요소로서의 Deeplearning4j의 역할이 [마이크로 서비스 아키텍처](http://microservices.io/patterns/microservices.html)에 적합한 최초의 딥 러닝 프레임 워크를 가능하게 합니다.

### DL4J의 신경망(Neural Networks)

* 제한 볼츠만 머신([Restricted Boltzmann machines](../restrictedboltzmannmachine.html))
* 합성곱 망([Convolutional Nets](../convolutionalnets.html)) (이미지)
* 순환 망([Recurrent Nets)/LSTMs](../recurrentnetwork.html) (시계열 및 센서 데이터)
* 재귀 오토인코더([Recursive autoencoders](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/feedforward/autoencoder/recursive/RecursiveAutoEncoder.java))
* 심층 신뢰 네트워크([Deep-belief networks](../deepbeliefnetwork.html))
* 딥 오토인코더([Deep autoencoders](http://deeplearning4j.org/deepautoencoder.html)) (QA/데이터 압축)
* 순환 뉴럴 텐서 네트워크(Recursive Neural Tensor Networks) (장면, 구문 분석)
* 누적된 잡음 제거용 오토인코더([Stacked Denoising Autoencoders](http://deeplearning4j.org/stackeddenoisingautoencoder.html))
* 더 많은 정보를 위해서는 "[신경망을 선택하는 방법](http://deeplearning4j.org/neuralnetworktable.html)"을 참조하십시오.

심층 신경망은 [기록적인 정확성](http://deeplearning4j.org/accuracy.html)을 가지고 있습니다. 신경망에 관한 간단한 소개는 저희의 [개요](http://deeplearning4j.org/neuralnet-overview.html) 페이지에 있습니다. 간단히 말해서, Deeplearning4j는 소위 레이어(layer)라 불리는 얕은(shallow)망으로부터 여러분이 심층(deep)망을 구성할 수 있게 합니다. 이러한 유연성은 여러분이 연동할 배포된 생산 수준의 프레임워크에서 요구되는 대로 제한 볼츠만 머신, 다른 오토인코더, 합성곱 망과 순환 망을 배포된 CPUs또는 GPUs 상의 스파크와 하둡과 결합하게 합니다.

저희가 개발한 다른 라이브러리들과 이들이 내장된 더 큰 에코시스템의 개요는 다음과 같습니다.

![Alt text](../img/schematic_overview.png)
 
여러분이 딥 러닝 네트워크를 학습할 때 조정해야 할 많은 매개 파라미터들이 있습니다. 저희가 이들을 설명하기 위해 최선을 다한 결과, Deeplearning4j는 자바, [스칼라](https://github.com/deeplearning4j/nd4s) 및 [Clojure](https://github.com/whilo/clj-nd4j)의 프로그래머들에게 DIY 도구로서의 역할을 다 할 수 있습니다.

질문이 있다면 [Gitter](https://gitter.im/deeplearning4j/deeplearning4j)를 통해; 고급지원을 원하시면 [Skymind](http://www.skymind.io/contact/)를 통해 연락주시기 바랍니다. [ND4J는 저희의 상위체계 연산에 동력을 지원하는 자바 기반 과학 컴퓨팅 엔진입니다](http://nd4j.org/). 큰 매트릭스들 상에서, 저희의 벤치마크는 [ND4J가 Numpy보다 대략 두 배 정도 빠르게 실행](http://nd4j.org/benchmarking) 함을 보여줍니다.

### Deeplearning4j 후기
프랑켄슈타인이 된 것 같았다. 그 의사..." - 스티브 D.

![Alt text](../img/logos_8.png)
