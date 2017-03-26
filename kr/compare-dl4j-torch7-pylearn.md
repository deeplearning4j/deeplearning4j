---
title: "프레임 워크 비교 : Deeplearning4j, Torch, Theano, TensorFlow, Caffe, Paddle, MxNet, Keras 및 CNTK"
layout: kr-default
redirect_from: kr/kr-compare-dl4j-torch7-pylearn
---

# 프레임 워크 비교 : Deeplearning4j, Torch, Theano, TensorFlow, Caffe, Paddle, MxNet, Keras 및 CNTK

Deeplearning4j는 최초의 오픈 소스 딥러닝 프로젝트는 아니지만 프로그래밍 언어와 의도 모두 과거 프레임 워크와 차별화됩니다. DL4J는 JVM 을 기반으로 하는 산업 중심의 상업적 지원을 하는 **분산 딥러닝 프레임워크** 로서 막대한 양의 데이터가 빠른 속도로 문제를 해결하도록 하였습니다. DL4J는 다수의 [CPUs](./native)와 [GPUs](./gpu) 를 사용하는 Hadoop 및 [Spark](./spark) 와 통합되며, 문제가 발생할 경우 [홈페이지를 통해 연락](http://www.skymind.ai/contact)이 가능합니다.

DL4J는 AWS, Azure 또는 Google Cloud와 같은 클라우드 서비스에 종속적이지 않고 용이하며 플랫폼 중립적입니다. 속도 측면에서는 여러개의 GPU를 사용한 복잡한 이미지 처리 작업에서는 [Caffe와 동일](https://github.com/deeplearning4j/dl4j-benchmark)하며, Tensoreflow 또는 Torch 보다는 우수합니다. Deeplearning4j 벤치마킹에 대한 자세한 내용은 [벤치마크 페이지](https://deeplearning4j.org/benchmark)를 참조하십시오. 이 페이지에서는 JVM의 힙 스페이스(heap space), GC(Garbage Collection) 알고리즘, 메모리 관리 및 DL4J의 ETL파이프라인을 조정하여 성능을 최적화할 수 있습니다. Deeplearning4j는 Java, [Scala](https://github.com/deeplearning4j/scalnet) and [Python APIs, the latter using Keras](./keras)를 기반하고 있습니다.

<p align="center">
<a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">DEEPLEARNING4J 시작하기</a>
</p>

### Content

Lua (루아)

* <a href="#torch">Torch & Pytorch</a>

파이썬 프레임워크

* <a href="#theano">Theano & 생태계</a>
* <a href="#tensorflow">TensorFlow</a>
* <a href="#caffe">Caffe</a>
* <a href="#chainer">Chainer</a>
* <a href="#cntk">CNTK</a>
* <a href="#dsstne">DSSTNE</a>
* <a href="#dsstne">DyNet</a>
* <a href="#keras">Keras</a>
* <a href="#mxnet">Mxnet</a>
* <a href="#paddle">Paddle</a>
* <a href="#bigdl">BigDL</a>
* <a href="#licensing">라이센싱</a>

JVM 고려사항

* <a href="#speed">속도</a>
* <a href="#java">DL4J: 왜 JVM인가?</a>
* <a href="#ecosystem">DL4J: 생태계</a>
* <a href="#scala">DL4S: Scala기반 딥러닝</a>
* <a href="#ml">머신러닝 프레임워크</a>
* <a href="#tutorial">더 보기</a>

## Lua

### <a name="torch">Torch & Pytorch</a>

[**Torch**](http://torch.ch/)는 Lua언어로 작성된 API를 제공하는 계산 프레임 워크로 머신러닝 알고리즘을 지원합니다. Torch의 어떤 버전들은 페이스 북이나 트위터와 같은 대형 기술 회사에서 자체 버전을 개발하여 사용하고 있습니다. Lua는 다중 패러다임 스크립팅 언어로 1990 년대 초 브라질에서 개발되었습니다.

Torch7은 많은 장점이 있지만 파이썬 기반의 학계나 자바 언어를 사용하는 소프트웨어 엔지니어에게 [쉽게 접근할 수 있도록 설계되지는 않았습니다](https://news.ycombinator.com/item?id=7929216). 반면에 Deeplearning4j는 업계에서의 사용 편의성을 반영하여 자바로 작성되었습니다. 우리는 상용성이 보다 광범위한 딥러닝 구현을 방해하는 요소라고 믿습니다. 우리는 하둡이나 스파크처럼 오픈 소스 분산 런타임을 통해 확장성을 자동화해야 한다고 생각합니다. 또한 상업적 지원이 가능한 오픈 소스 프레임워크야말로 작업 도구(working tools)를 보장하고 커뮤니티를 구축하는 데 가장 적합한 솔루션이라고 생각합니다.

[Pytorch](https://github.com/pytorch/pytorch)로 알려진 Torch용 Python API는 2017 년 1 월 Facebook에서 오픈 소스화되었습니다.  PyTorch는 가변 길이 입력 및 출력을 처리 할 수 있는 다이나믹 컴퓨테이션 그래프를 제공하며 특히 RNN을 사용할 때 유용합니다. 다이나믹 컴퓨테이션 그래프를 지원하는 다른 프레임 워크로는 CMU의 DyNet과 PFN 's Chainer가 있습니다.

장점과 단점:

* (+) 결합하기 쉬운 많은 모듈 조각
* (+) 자신만의 레이어 유형을 작성하고 GPU에서 실행하기 쉬움
* (+) Lua. ;) (대부분의 라이브러리 코드는 Lua로 되어있음, 읽기 쉬움)
* (+) 사전 학습된 모델 대거 존재
* (+) PyTorch
* (-) Lua
* (-) 보통 스스로 학습 코드를 작성해야 함 (적은 플러그 앤 플레이)
* (-) 상업적 지원 없음
* (-) 미완성 문서

## 파이썬 프레임워크

### <a name="theano">Theano와 생태계</a>

많은 딥러닝 분야의 학술 연구자들은 [파이썬](http://darkf.github.io/posts/problems-i-have-with-python.html)으로 작성된 딥러닝 프레임 워크의 대부로 불리는 [**Theano**](http://deeplearning.net/software/theano/)에 의존합니다.
Theano는 Numpy와 같은 다차원 배열을 다루는 라이브러리입니다.  Theano는 데이터 탐색에 적합하며 다른 라이브러리와 함께 연구용으로 개발되어 사용됩니다.

[Keras](https://github.com/fchollet/keras),  [Lasagne](https://lasagne.readthedocs.org/en/latest/), [Blocks](https://github.com/mila-udem/blocks) 등 수많은 오픈 소스 딥러닝 라이브러리가 Theano를 기반으로 탄생했습니다. 이 라이브러리들은 Theano의 일부 비직관적인 인터페이스 위에 API를 사용하기 쉽게 레이어를 추가하는 방식입니다. (2016 년 3 월 현재 [Pylearn2는 더이상 유효하지 않은 것](https://github.com/lisa-lab/pylearn2) 같습니다.)

반면에 Deeplearning4j는 JVM언어인 Java 및 Scala로  솔루션을 생성하며 딥러닝을 프로덕션에 최적화 하였습니다. 병렬 GPU나 CPU에서 가능한 한 많은 노브(knobs)를 자동화하고 확장 가능한 방식(Scalable Fashion)으로 필요에 따라 Hadoop 및 [Spark](./spark.html)와 통합하는 것을 목표로 합니다.

장점과 단점

* (+) Python + Numpy
* (+) 컴퓨테이셔널 그래프에 적합한 추상화
* (+) 컴퓨테이셔널 그래프에 RNN이 잘 맞음.
* (-) Raw Theano는 수준이 낮은 편임.
* (+) 고급 래퍼 (Keras, Lasagne) 의 어려운 부분을 완화
* (-) 오류 메시지는 도움이 되지 않는 경우가 있음.
* (-) 대형 모델은 컴파일 시간이 오래 걸릴 수 있음.
* (-) 토치보다 훨씬 "복잡함"
* (-) 사전 학습 된 모델에 대한 패치 지원
* (-) AWS에서 잦은 버그가 발생
* (-) 단일 GPU

### <a name="tensorflow">TensorFlow</a>

* Google은 Theano를 대체하고자 TensorFlow를 만들었습니다. 사실 두 라이브러리는 아주 비슷합니다. Ian Goodfellow와 같은 Theano의 창시자 중 일부는 OpenAI로 떠나기 전에 Google에서 Tensorflow를 만들었습니다.
*	현재 **TensorFlow** 는 소위 "인라인" 행렬 연산을 지원하지 않지만 실행을 위해서는 행렬을 복사해야 합니다. 매우 큰 행렬을 복사하는 것은 모든 면에서 비용이 많이 듭니다. Tensorflow는 최신 딥러닝 도구에 비해 네 배나 시간이 더 걸립니다. 구글은 이 문제에 대해 연구하고 있다고 이야기 합니다.
* 대부분의 딥러닝 프레임 워크와 마찬가지로 TensorFlow는 C / C ++ 엔진에 Python API로 작성되어 빠른 실행이 가능합니다. Java 및 Scala  커뮤니티를 위한 것은 아닙니다.
* TensorFlow 는 CNTK와 같은 [다른 프레임워크 보다 속도면에서 많이 느립니다](https://arxiv.org/pdf/1608.07249v7.pdf).
* TensorFlow는 딥러닝 이상의 것입니다. TensorFlow에는 실제로 강화학습(Reinforcement Learning) 및 기타 알고리즘을 지원하는 도구가 있습니다.
* Tensorflow가 인정한 Google의 목표는 연구원들이 짠 코드를 공유하고 소프트웨어 엔지니어가 딥러닝에 접근하는 방법을 표준화하며 TensorFlow가 최적화 된 Google Cloud 서비스에 대한 추가적인 그리기를 창출하는 것입니다.
* TensorFlow는 상업적 지원이 되지 않으며 향후에도 Google은 엔터프라이즈를 위한 오픈 소스 소프트웨어 지원사업에 뛰어들 것 같지 않습니다. 현재는 그저 연구원들에게 툴을 제공할 뿐입니다.
* Theano와 마찬가지로 TensforFlow는 컴퓨테이션  그래프 (예 : z = sigmoid (x)와 같이 x와 z가 행렬인 일련의 행렬 연산)를 생성하고 자동으로 미분합니다. 자동 미분이 중요한 이유는 신경망을 업데이트 시키기 위한 역전파(Backpropagation) 변화량을 매번 수동으로 계산하고 반영할 필요가 없기 때문입니다. Google의 생태계에서 컴퓨테이션 그래프는 구글 브레인 (Google Brain)이중추적인 역할을 하였지만  아직 오픈 소스화 하지는 않았습니다. Google의 사내 딥러닝 솔루션의 절반은 Tensorflow로 구축되어 있습니다.
* 기업의 관점에서 볼 때, 생각해봐야 할 문제는 이러한 도구를 Google에 전적으로 의존하여 사용해야 하는가의 여부입니다.
*	주의 사항: Tensorflow의 모든 작업이 Numpy에서 하는 것처럼 작동하는 것은 아닙니다.

장점과 단점

* (+) Python + Numpy
* (+) 컴퓨테이션 그래프 앱스트랙션, Theano와 비슷
* (+) Theano보다 컴파일 시간이 빠름
* (+) TensorBoard 시각화
* (+) 데이터 및 모델 병렬 처리
* (-) 다른 프레임 워크보다 느림.
* (-) Torch보다 훨씬 “더 복잡”하지만 기능이 많음.
* (-) 사전 학습된 모델 부족.
* (-) 컴퓨테이션 그래프는 순수 파이썬으로 속도가 느림.
* (-) 상업적 지원 없음.
* (-) 파이썬으로 드롭 아웃되어 각각의 새로운 트레이닝 배치를 로드
* (-) 툴로 사용하기 어려움.
* (-) 다이나믹 타이핑은 대형 소프트웨어 프로젝트에서 오류가 발생하기 쉬움.

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/)는 머신 비전 라이브러리로  Matlab이 C 및 C ++에 고속 컨볼루션 신경망을 구현한 것으로서 널리 사용되고 있습니다. (속도와 기술적인 빚(Technical Debt)의 트레이드오프를 알고 싶다면 [칩에서 칩으로 C ++ 포팅에 대한 Steve Yegge의 이야기](https://sites.google.com/site/steveyegge2/google-at-delphi))를 참조할 것). Caffe는 문자, 음성, 또는 시계열 데이터와 같은 다른 딥러닝 응용 프로그램을 위한 것이 아닙니다. 여기 언급된 다른 프레임 워크와 마찬가지로 Caffe는 API로 Python을 선택했습니다.

Deeplearning4j와 Caffe는 모두 최신의 컨볼루션 신경망을 사용하여 이미지를 분류합니다. Caffe와 달리 Deeplearning4j는 임의의 수의 칩에 대해 병렬 GPU를 *지원* 할뿐만 아니라 여러 GPU 클러스터에서 딥러닝을 보다 원활하게 진행할 수있는 많은 기능을 제공합니다. Caffe는, 논문에서 광범위하게 인용되었지만, 주로 Model Zoo 사이트에서 호스팅되는 사전 학습 모델의 소스로 사용됩니다. Deeplearning4j는 Caffe 모델을 Spark로 가져오기 위한 [파서(parser)](https://github.com/deeplearning4j/deeplearning4j/pull/480)를 구축하고 있습니다.

장점과 단점

* (+) 피드 포워드 네트워크 및 이미지 처리에 적합.
* (+) 기존 네트워크 미세 조정에 적합.
* (+) 코드 작성 없이 모델 트레이닝 가능.
* (+) 파이썬 인터페이스가 매우 유용.
* (-) 새로운 GPU 레이어에 C ++ / CUDA를 작성해야 함.
* (-) 회귀망에 적합하지 않음
* (-) 큰 네트워크를 다루기에는 불편함 (GoogLeNet, ResNet)
* (-) 확장성이 없음
* (-) 상업적 지원 없음
* (-) 개발이 느려지고 있으며, 곧 유효하지 않을 수 있음

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK)는 Microsoft의 오픈 소스 딥러닝 프레임 워크입니다. Computational Network Toolkit"의 약자이며, 피드 포워드 DNN, 컨볼루션 넷 및 회귀망이 포함됩니다. CNTK는 C ++ 코드로 Python API를 제공합니다. CNTK는 [허용 라이센스](https://github.com/Microsoft/CNTK/blob/master/LICENSE.md)가있는 것처럼 보이지만 놀랍게도ASF 2.0, BSD 또는 MIT와 같은 일반적인 라이선스 중 하나를 채택하지는 않았습니다. 이 라이센스는 CNTK가 분산 교육을 쉽게 수행 할 수있는 방법 (1 bit SGD)에 적용되지 않으며, 상업적 용도로 부여되지도 않았습니다.

### <a name="chainer">Chainer</a>

CChainer는 Python API를 제공하는 오픈소스 신경망 프레임워크로, 개발자들 중 코어팀이 [Preferred Networks](https://www.crunchbase.com/organization/preferred-networks#/entity)에서 일하고 있습니다. Preferred Networks는 도쿄에 기반을 두고 있는 기계학습 스타트업이며 대부분의 엔지니어들이 도쿄대학 출신입니다.  Chainer는 CMU의 DyNet과 Facebook의 PyTorch가 출현하기 전까지 다이나믹 컴퓨테이션 그래프나 가변 길이의 입력을 지원하는 신경망으로 NLP 작업에 많이 사용되는 최고의 프레임워크였습니다. [벤치마크](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html)에 따르면 Chainer는 다른 Python 기반 프레임워크보다 눈에 띄게 빠릅니다 TensorFlow는 MxNet 및 CNTK를 포함하는 테스트 그룹 중 가장 느린 속도를 보입니다.

### <a name="dsstne">DSSTNE</a>

Amazon의 Deep Scalable Sparse Tensor Network Engine ([DSSTNE](https://github.com/amznlabs/amazon-dsstne))은 머신러닝 및 딥러닝 모델 구축을 위한 라이브러리입니다. Tensorflow와 CNTK 이후 출시 된 많은 오픈 소스 딥러닝 라이브러리 중 가장 최근의 것이라고 볼 수 있습니다. Amazon은 AWS로 MxNet을 지원했기 때문에 미래가 명확하지 않지만, C ++로 작성된 DSSTNE는 속도 면에서 우수하다고 할 수 있습니다. 단, 다른 라이브러리만큼 큰 커뮤니티가 구축된 것은 아닙니다.

* (+) 스파스 인코딩 처리
* (-) Amazon은 최상의 결과를 얻는데 필요한 [모든 정보를 예제로 공유하지 않았을 수 있음](https://github.com/amznlabs/amazon-dsstne/issues/24).
* (-) Amazon은 AWS에서 사용할 다른 프레임 워크를 외부에서 선택함.

### <a name="dynet">DyNet</a>

[Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980) 인 [DyNet](https://github.com/clab/dynet)은 Carnegie Mellon University에서 나왔으며 cnn이라고 불려 왔습니다. 주목할만한 특징은 NLP에 적합한 가변 길이의 입력을 지원하는 다이나믹 컴퓨테이션 그래프입니다. PyTorch와 Chainer도 같은 것을 제공합니다

* (+) 다이나믹 컴퓨테이션 그래프
* (-) 작은 사용자 커뮤니티

### <a name="keras">Keras</a>

[Keras](keras.io)는 Theano와 TensorFlow를 백엔드(back-end)로 사용하는딥러닝 라이브러리로서 Torch에서 영감을 얻어 직관적 API를 제공합니다. 아마도 이것은 현존하는 최고의 파이썬 API 일 것입니다. Deeplearning4j는 [Keras를 통해 Theano와 Tensorflow에서 혹은 Keras자체의 모델을 가져옵니다](./model-import-keras). 창시자는 Google의 소프트웨어 엔지니어인 [Francois Chollet](https://twitter.com/fchollet)입니다.


* (+) Torch에서 영감을 얻은 직관적 API
* (+) Theano와 작업 가능, TensorFlow와 Deeplearning4j 백엔드 (CNTK 백엔드 예정)
* (+) 빠르게 성장하고 있는 프레임워크
* (+) 신경망의 표준 Python API이 될 가능성이 큼

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet)은 [Amazon Web Services](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)에서 채택한 R, Python 및 Julia와 같은 언어를 API로 사용하는 머신러닝 프레임 워크입니다. 애플의 일부가 2016 년에 Graphlab / Dato / Turi가 인수된 후에도 사용하고 있다는 소문이 돌고 있습니다. MxNet은 빠르고 유연하며 현재 Pedro Domingos와 워싱턴 대학 연구원 팀이 참여하고 있습니다. MxNet과 Deeplearning4j의 장단점 [비교](https://deeplearning4j.org/mxnet)는 여기에서 확인할 수 있습니다.

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle)은 [Baidu가 만들고 지원](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html)한 딥러닝 프레임 워크로, “PArallel Distributed DeepLEarning”의 약자입니다. Paddle은 출시된 주요 프레임 워크 중 가장 최근의 것이며 다른 프레임 워크와 마찬가지로 Python API를 제공합니다.

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL)은 Apache Spark에 초점을 맞춘 새로운 딥러닝 프레임 워크로서 Intel 칩에서만 작동합니다.

### <a name="licensing">라이센싱</a>

각각의 오픈소스들의 라이센싱 정책은 다음과 같습니다. Theano, Torch 및 Caffe는 특허 및 특허 분쟁을 다루지 않는 BSD 라이센스를 사용합니다. Deeplearning4j 및 ND4J는 **[Apache 2.0 라이센스](http://en.swpat.org/wiki/Patent_clauses_in_software_licences#Apache_License_2.0)** 하에 배포되며 특허권 부여 및 피소송 조항을 모두 포함합니다. 즉, 누구나 자유롭게 Apache 2.0 라이센스 코드를 기반으로 한 파생물을 만들 수 있지만 원래 코드 (이 경우 DL4J)와 관련하여 다른 사람이 특허 청구를 제기하면 즉시 모든 특허 청구를 잃게됩니다. (즉, 소송에서 자신을 지키기 위한 자원은 제공되지만 다른 사람들을 공격하는 데에는 사용될 수 없습니다.) BSD는 일반적으로이 문제는 다루지 않습니다.

## JVM 고려사항

### <a name="speed">속도</a>

ND4J로 수행되는 Deeplearning4j의 기본 선형 대수 연산은 매우 큰 행렬 곱셈에서 [Numpy보다 적어도 두 배 빠른 속도로 실행](http://nd4j.org/benchmarking)되는 것으로 나타났습니다. 그것이 우리의 프레임 워크를 NASA의 제트 추진 연구소 (Jet Propulsion Laboratory)의 팀이 채택한 이유 가운데 하나입니다. 또한 Deeplearning4j는 CUDA C를 사용하는 x86 및 GPU를 포함한 다양한 칩에서 실행되도록 최적화하였습니다.

Torch7과 DL4J는 모두 병렬 처리를 사용하지만 DL4J의 **병렬 처리는 자동** 입니다. 즉, 작업자 노드의 연결 설정을 자동화하여 [Spark](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark), [Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn), 또는 [Akka 및 AWS](http://deeplearning4j.org/scaleout.html)에서 대규모 병렬 네트워크를 생성하면서 사용자가 라이브러리들을 우회 할 수 있도록 합니다. Deeplearning4j는 특정 문제를 해결하고 신속하게 처리하는 데 가장 적합합니다.

Deeplearning4j의 전체 기능 목록은 [기능 페이지](./features.html)를 참조하십시오.

### <a name="java">왜 JVM인가?</a>

우리는 종종, 왜 JVM을 기반으로 오픈소스 딥러닝 프로젝트를 구현하였는지 질문을 받곤 합니다. 파이썬에는 결국 자바처럼 명시적인 클래스(explicit classes)를 작성하지 않고 행렬을 함께 추가할 수있는 훌륭한 구문 요소(synthetic elements)가 있기 때문입니다. 마찬가지로 Python은 Theano와 Numpy와 같은 기본 확장 기능을 갖춘 방대한 과학 컴퓨팅 환경을 갖추고 있습니다.

그러나 JVM과 그 주요 언어인 Java와 Scala에는 몇 가지 추가적인 장점이 있습니다.

첫째, 대부분의 대기업과 정부 기관은 Java나 JVM 기반 시스템에 크게 의존하고 있습니다. 그들은 JVM 기반 AI에 함께 막대한 투자를했습니다. Java는 엔터프라이즈에서 가장 널리 사용되는 언어로서 Hadoop, ElasticSearch, Hive, Lucene 및 Pig의 언어이며 머신러닝 문제에 유용합니다. Spark와 Kafka는 또 다른 JVM 언어인 스칼라로 작성되었습니다. 즉, 현실에서 문제를 해결하는 대다수의 프로그래머들에게 언어 장벽이 없이 딥러닝의 이점 활용이 가능합니다. 우리는 딥러닝을 즉각적으로 사용할 수 있는 잠재 고객에게 한층 유용한 활용수단이 되고자 합니다. 1 천만 개발자가 있는 Java는 세계에서 가장 큰 프로그래밍 언어입니다.

둘째, 자바와 스칼라는 본질적으로 파이썬보다 빠릅니다. Cython에 대한 의존도를 무시하고 Python으로만 작성하면 결국 느려질 것입니다. 계산 상 비싼 연산(Computationally expensive operations)은 C 또는 C ++로 작성됩니다. (우리가 작업에 관해서 이야기 할 때, 우리는 문자행렬이나 고수준의 머신러닝 과정과 관련된 다른 작업도 고려합니다.) 초기에 파이썬으로 작성된 대부분의 딥러닝 프로젝트는 제작 단계로 넘어갈 때 처음부터 모두 다시 작성해야만 합니다 . Deeplearning4j는 [JavaCPP](https://github.com/bytedeco/javacpp)를 사용하여 사전 컴파일 된 네이티브 C ++을 Java에서 호출하여 트레이닝 속도를 크게 향상시킵니다. 대부분의 Python 프로그래머는 Scala에서 딥러닝을 선택하는데, 공유된 코드를 기반으로 다른 사람들과 작업을 할 때 정적 유형 지정(static typing) 및 함수 프로그래밍을 선호하기 때문입니다.

셋째, Java의 강력한 과학 컴퓨팅 라이브러리에서 부족한 점은 작성한 후 이를 [ND4J](http://nd4j.org)로 처리하여 해결할 수 있습니다. ND4J는 분산 된 GPU 또는 GPU에서 실행되며 Java 또는 Scala API를 통해 인터페이스 될 수 있습니다.

마지막으로 Java는 Linux 서버, Windows 및 OSX 데스크탑, Android 폰 및 임베디드 자바를 통한 사물들의 인터넷 메모리 센서에서 연동되어 작동하는 안전한 네트워크 언어입니다. Torch 및 Pylearn2는 C ++을 통해 최적화하고 유지 보수를 시도하는 사람들이 어려움을 겪지만 Java는 "어느 한 곳에서 작성하여 어디에서나 쓸 수 있는" 언어로서 많은 플랫폼에서 딥러닝을 사용해야 하는 기업에 적합합니다.

### <a name="ecosystem">생태계</a>

Java의 인기는 생태계에 의해 더욱 강화됩니다. Hadoop은 Java로 구현됩니다. [Spark](https://spark.apache.org/)는 [Hadoop](https://hadoop.apache.org/) 의 Yarn 런타임에서 실행됩니다. [Akka](https://www.typesafe.com/community/core-projects/akka)와 같은 라이브러리는 Deeplearning4j를 위한 분산 시스템을 구현할 수 있도록 만들었습니다. 다시 말해, Java는 거의 모든 응용 프로그램에 대해 고도의 인프라를 자랑하며 Java로 작성된 딥러닝 넷은 데이터 가까이 존재하기 때문에 프로그래머의 삶을 편하게 만들어줍니다. Deeplearning4j는 YARN 앱으로 실행 및 프로비저닝 할 수 있습니다.

Java는 기본적으로 Scala, Clojure, Python 및 Ruby와 같은 다른 널리 사용되는 언어에서도 사용할 수 있습니다. Java를 선택함으로써 우리는 주요 프로그래밍 커뮤니티를 최소화시킬 수 있었습니다.

Java가 C 나 C ++만큼 빠르지는 않지만 많은 사람들이 생각하는 것보다는 훨씬 빠르며 우리는 GPU 이건 CPU 이건간에 더 많은 노드를 추가하여 가속화 할 수있는 분산 시스템을 구축했습니다. 즉, 속도를 원하면 더 많은 서버를 추가하시면 됩니다.

마지막으로 우리는 DL4J 용 Java에서 ND-Array를 포함하여 Numpy의 기본 응용 프로그램을 개발하고 있습니다. 우리는 Java의 많은 단점을 신속하게 해결할 수 있다고 믿으며 많은 이점들은 지속해 나갈 것입니다.

### <a name="scala">Scala</a>

Deeplearning4j와 ND4J를 구축 할 때 우리는 [Scala](./scala)에 특별한 관심을 기울였습니다. 스칼라가 데이터 사이언스에서 지배적인 언어가 될 잠재력이 충분히 있다고 믿기 때문입니다. [Scala API](http://nd4j.org/scala.html)를 사용하여 JVM에 대한 수치 연산, 벡터화 및 딥러닝 라이브러리를 작성하면 커뮤니티가 목표를 향해 자연스레 이동하게 됩니다.

DL4J와 다른 프레임 워크의 차이점을 실제로 이해하려면 DL4J를 [직접 시험](./quickstart)해 보십시오.

### <a name="ml">머신러닝 프레임워크</a>

위에 열거 된 딥러닝 프레임워크는 일반적인 머신러닝 프레임워크보다 심화되어 있습니다. 우리는 여기 주요 프레임 워크를 나열 할 것입니다 :

* [sci-kit learn](http://scikit-learn.org/stable/) - Python을 위한 기본 오픈 소스 머신러닝 프레임워크입니다.
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html) - Apache의 핵심 머신러닝 프레임워크입니다. Mahout은 분류, 클러스터링 및 권장 (recommendations)을 수행합니다.
* [SystemML](https://sparktc.github.io/systemml/quick-start-guide.html) - IBM의 머신러닝 프레임워크로 기술 통계(Descriptive Statistics), 분류, 클러스터링, 회귀, 행렬 인수 분해 및 생존 분석을 수행하고 지원 벡터 시스템을 포함합니다.
* [Microsoft DMTK](http://www.dmtk.io/) - Microsoft의 분산 시스템 학습 툴 키트입니다. 분산 단어 임베딩 및 LDA.

### <a name="tutorial">Deeplearning4j Tutorials</a>

* [심층 신경망 소개](./neuralnet-overview)
* [컴볼루셔널 네트워크 자습서](./convolutionalnets)
* [LSTM 및 회귀망 자습서](./lstm)
* [DL4J에서 회귀망 사용](./usingrnns)
* [MNIST를 사용한 심층 네트워크](./deepbeliefnetwork)
* [Canova를 사용하여 맞춤형 데이터 파이프라인 구축](./image-data-pipeline)
* [제한된 Boltzmann 기계](./restrictedboltzmannmachine)
* [고유벡터(Eigenvectors), PCA 및 엔트로피](./eigenvector.html)
* [딥러닝 용어집](./glossary.html)
* [Word2vec, Doc2vec & GloVe](./word2vec)
