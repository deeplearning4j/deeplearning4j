---
title: 초보자를 위한 MNIST
layout: kr-default
redirct_from: kr/kr-mnist-for-beginners
---

<header>
  <h1>초보자를 위한 MNIST</h1>
  <p>이 튜토리얼에서 머신러닝계의 “Hello World” 인 MNIST의 데이터 세트는 다음과 같습니다.</p>
  <ol class="toc">
    <li><a href="#introduction">MNIST 소개</a></li>
    <li><a href="#mnist-dataset">MNIST 데이터 세트</a></li>
    <li><a href="#configuring">MNIST 예제 구성</a></li>
    <li><a href="#building">신경망 구축</a></li>
    <li><a href="#training">모델 트레이닝</a></li>
    <li><a href="#evaluating">결과 평가</a></li>
    <li><a href="#conclusion">결론</a></li>
  </ol>
</header>
  <p>예상 소요시간은 30분 입니다.</p>
<section>
  <h2 id="MNIST 소개">Introduction</h2>
  <img src="/img/mnist_render.png"><br><br>
  <p>MNIST는 손으로 필기한 숫자 각각의 이미지가 정수로 표시되는 데이터베이스입니다. 머신러닝 알고리즘의 성능을 벤치마킹하는 데 사용됩니다. 딥러닝은 MNIST에서 매우 잘 수행되며, 99.7 % 이상의 정확도를 달성하였습니다.</p>
  <p>우리는 MNIST를 사용하여 각각의 이미지를 보고 그 숫자를 예측하기 위해 신경망을 트레이닝 할 것입니다. 그 첫 번째 단계는 바로 Deeplearning4j를 설치하는 것입니다.</p>
  <a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH DEEPLEARNING4J</a>
</section>

<section>
  <h2 id="mnist-dataset">MNIST 데이터 세트</h2>
  <p>MNIST 데이터 세트에는 6만건의 사례로 구성된 교육 세트와 1만건의 사례로 구성된 테스트 세트가 포함되어 있습니다. 트레이닝 세트는 올바른 레이블인 정수를 예측하는 알고리즘을 가르치기 위해 사용되며 테스트 세트는 트레이닝 된 네트워크가 얼마나 정확하게 추측 할 수 있는지 확인하는 데 사용됩니다.</p>
  <p>머신러닝의 세계에서 이것은 지도학습 <a href="https://en.wikipedia.org/wiki/Supervised_learning" target="_blank">supervised learning</a> 이라고 하는데, 우리가 만들어내고자 하는 상상적 이미지에 대해 정확히 추측된 답을 가지고 있기 때문입니다. 따라서 트레이닝 세트는 감독자 혹은 교사의 역할을 하게 되며 이것이  잘못된 추측을 할 경우엔 신경망을 시정할 수 있습니다.</p>
</section>

<section>
  <h2 id="configuring">예제 구성</h2>
  <p>MNIST 튜토리얼은 Maven에 패키징 되어있어 별도의 코드를 작성할 필요가 없습니다. IntelliJ를 열어 시작하십시오. (IntelliJ를 다운로드 하려면, <a href="quickstart">Quickstart…</a>)를 참조하십시오.</p>
  <p><code>dl4j-examples</code>폴더를 여십시오. <kbd>src</kbd> → <kbd>main</kbd> → <kbd>java</kbd> → <kbd>feedforward</kbd> → <kbd>mnist</kbd>디렉토리로 이동하여 <code>MLPMnistSingleLayerExample.java</code>파일을 엽니다.</p>
  <p><img src="/img/mlp_mnist_single_layer_example_setup.png"></p>
  <p>이 파일에서 우리는 신경망을 구성하고, 모델을 트레이닝 시키고 결과를 평가할 것입니다. 튜토리얼과 함께 이 코드를 보는 게 도움이 되겠습니다.</p>
  <h3>변수 설정</h3>
  <pre><code class="language-java">
    final int numRows = 28; // The number of rows of a matrix.
    final int numColumns = 28; // The number of columns of a matrix.
    int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
    int batchSize = 128; // How many examples to fetch with each step.
    int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later.
    int numEpochs = 15; // An epoch is a complete pass through a given dataset.
  </code></pre>
  <p>이 예제에서 각 MNIST 이미지는 28x28 픽셀입니다. 즉, 입력 데이터가 28 개의 numRow x 28 numColumns 행렬입니다 (행렬은 딥러닝의 기본 데이터 구조입니다). 또한, MNIST는 10 개의 가능한 결과 (0-9까지의  숫자가 메겨진 라벨)를 포함하고 있는데 이것이 바로<b>outputNum</b>입니다.</p>
  <p><b>batchSize</b> 및 <b>numEpochs</b>는 경험을 바탕으로 선택해야 합니다. 실험을 통해야만 무엇이 작동 하는지를 정확히 배울 수 있습니다. 더 큰 배치 사이즈(Batch size) 는 더 빠른 학습이 가능하며 더 많은 주기(Epoch)와 데이터 세트를 거치면서 정확도가 향상됩니다.</p>
  <p>그러나 일정 주기(Epoch)를 넘어서면 정확성은 감소하기 때문에 정확도와 교육 속도 간에는 트레이드 오프가 있습니다. 일반적으로 실험을 통해야 최적값을 발견할 수 있습니다. 이 예제에서는 우리가 적절한 기본값을 설정했습니다.</p>
  <h3>MNIST데이터 가져오기</h3>
  <pre><code class="language-java">
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
  </code></pre>
  <p><code>DatasetIterator</code> 클래스는 MNIST 데이터 집합을 가져 오는 데 사용됩니다. 데이터 세트 <code>mnistTrain</code>은 모델 트레이닝을 위해, <code>mnistTest</code>은 트레이닝 후 <b>모델의 정확성을 평가</b>하기 위해 작성합니다. 여기서의 모델이란 신경망의 매개 변수를 의미합니다. 이 매개 변수는 입력 신호를 처리하는 계수로서 신경망이 학습할 때 각각의 이미지에 맞는 레이블을 결과 도출해낼 수 있을 때까지 조정됩니다. 이 때 당신은 정확한 모델을 가지게 됩니다.</p>
</section>

<section>
  <h2 id="building">신경망 구축</h2>
  <p>Xavier Glorot와 Yoshua Bengio의 논문<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank"></a>을 기반으로 우리는 피드 포워드(feedforward) 신경망을 구축할 예정입니다. 단일 히든 레이어로 기본 예제를 시작합니다. 그러나 일반적으로 신경망이 깊어질수록 (즉, 레이어가 많을수록) 복잡하고 미묘한 차이를 잡아내어 정확한 결과를 얻게 됩니다.</p>
  <img src="/img/onelayer.png"><br><br>
  <p>이 도표를 잊지 마십시오. 이것이 바로 우리가 구축하고있는 단일 레이어 신경망입니다.</p>
  <h3>하이퍼 파라미터 설정</h3>
  <p>Deeplearning4j로 작성된 신경망은 <a href="http://deeplearning4j.org/neuralnet-configuration.html" target="_blank">NeuralNetConfiguration class</a>.에 기초해 있습니다. 여기서 바로 신경망 구조와 어떻게 알고리즘을 학습하는가를 결정하는 하이퍼파라미터를 설정합니다.  직관적으로 각각의 하이퍼 파라미터는 음식의 구성성분처럼  음식이 아주 잘되거나 잘못 될 수 있는 것에 비유할 수 있습니다. 한 가지 다행인 것은 하이퍼 파라미터의 조정이 가능하여 올바른 결과를 도출해내도록 설정할 수 있습니다.</p>
  <pre><code class="language-java">
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
  </code></pre>
  <h5>.seed(rngSeed)</h5>
  <p>이 매개 변수는 임의로 생성 된 특정 웨이트(weight) 초기화를 사용합니다. 예제를 여러 번 실행하고 시작할 때마다 새로운 무작위 웨이트 초기값을 생성하면 신경망의 결과 - 정확도와 F1 점수 -가 매우 달라질 수 있습니다. 초기 웨이트가 다르면 알고리즘이 에러공간(errorscape)에서 다른 로컬미니마로 이끌 수 있기 때문입니다. 동일한 임의의 웨이트를 유지하면 다른 조건이 동일하게 유지되는 동안 다른 하이퍼 파라미터의 조정 효과를 보다 명확하게 분리 할 수 ​​있습니다.</p>
  <h5>.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)</h5>
  <p>SGD (Stochastic gradient descent)는 비용 함수를 최적화하는 일반적인 방법입니다. 자세한 내용은 <a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng의 Machine Learning 과정</a>과 SGD 용어 정의<a href="http://deeplearning4j.org/glossary#stochasticgradientdescent" target="_blank">glossary</a>를 참조하십시오. 오류를 최소화 할 수 있는 SGD 및 기타 최적화 알고리즘에 대한 내용을 배울 수 있습니다.</p>
  <h5>.iterations(1)</h5>
  <p>각각의 이터레이션은 신경망에 대한 학습 단계 입니다. 다시 말해 모델 웨이트의 업데이트 주기를 의미 합니다.신경망은 데이터에 노출되어 데이터에 대한 추측을 하며 추측 오류에 근거하여 자체 매개 변수를 수정합니다. 반복이 많을수록 신경망은 더 많은 단계를 거치며 더 많은 것을 배워 오류를 최소화 할 수 있습니다.</p>
  <h5>.learningRate(0.006)</h5>
  <p>여기에서는 러닝 레이트를 설정합니다. 러닝레이트는 각각의 이터레이션마다 웨이트의 조정 크기를 의미하는데 스텝 사이즈라고도 합니다. 큰 러닝레이트를 사용하면 신경망이 에러공간을 너무 빠르게 탐색하여 최소 오류지점을 찾지 못할 수 있습니다. 반면에 낮은 러닝 레이트를 사용하면 오류를 최소화 할 수는 있지만  웨이트를 조정할 때 조정 크기가 너무 작아서 매우 느리게 처리됩니다.</p>
  <h5>.updater(Updater.NESTEROVS).momentum(0.9)</h5>
  <p>모멘텀은 최적화 알고리즘으로 최적의 지점에 얼마나 빨리 도달하게 할지를 결정하는 부가적인 요소입니다. 또한 웨이트 조정방향에 영향을 미치기 때문에 코드에서는 웨이트 <code>업데이터</code>라고 볼 수 있습니다.</p>
  <h5>.regularization(true).l2(1e-4)</h5>
  <p>정형화 (regularization) 란 <b>오버피팅(overfitting)</b>을 방지하는 기술입니다. 오버피팅이란 모델과 트레이닝 데이터는 잘 맞추어졌으나 새로운 데이터에 노출되면 실제로는 제대로 수행되지 않는 경우를 말합니다.</p>
  <p>우리는 L2정형화를 사용하여, 개별 웨이트가 전반적인 결과에 과도한 영향을 미치는 것을 방지합니다.</p>
  <h5>.list()</h5>
  <p>리스트는 신경망의 레이어 수를 지정합니다. 이 기능은 구성(configuration)을 n 번 복제하고 레이어별로 구성할 수 있게 해줍니다.</p>
  <p>이해가 어렵다면  <a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng의 Machine Learning 과정</a>을 참조하십시오.</p>
  <h3>Building Layers</h3>
  <p>우리는 각각의 하이퍼 파라미터에 대한 연구 (예 : activation, weightInit)에 대해서는 다루지 않으려 합니다. 하이퍼 파라미터의 역할에 대해 간단히 정의할 것입니다. 하지만 왜 이것이 중요한지 더 알고 싶다면 <a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank">Xavier Glorot와 Yoshua Bengio의 논문</a>을 참조하십시오.</p>
  <img src="/img/onelayer_labeled.png"><br>
  <pre><code class="language-java">
    .layer(0, new DenseLayer.Builder()
            .nIn(numRows * numColumns) // Number of input datapoints.
            .nOut(1000) // Number of output datapoints.
            .activation("relu") // Activation function.
            .weightInit(WeightInit.XAVIER) // Weight initialization.
            .build())
    .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(1000)
            .nOut(outputNum)
            .activation("softmax")
            .weightInit(WeightInit.XAVIER)
            .build())
    .pretrain(false).backprop(true)
    .build();
  </code></pre>
  <p>히든 레이어란?</p>
  <p>히든 레이어의 각 노드 (원들)는 MNIST 데이터 세트에서 필기된 숫자의 특징을 의미합니다. 예를 들어, 숫자 6을 보고 있다고 생각해 보십시오. 한 노드는 둥근 모서리일 수 있고 다른 노드는 구부러진 선일 수 있습니다. 이러한 특징은 모델의 계수에 의해 중요도에 따라 가중치가 적용되며 각 히든 레이어에서 재조합되어 필기된 숫자가 실제로 6인지의 여부를 판단하는 데 도움이 됩니다. 더 많은 노드 레이어가 있다면 한층 복잡하고 미묘한 차이를 잡아낼 수 있게 되어 예측도를 향상시킬 수 있습니다.</p>
  <p>레이어가 ‘숨겨진’ 것이라고 생각할 수도 있습니다. 데이터가 신경망을 통해 결과를 도출해내는 것은 알고 있지만, 내부적으로 어떻게 데이터를 처리하기 때문인지는 아직 해독이 어렵기 때문입니다. 신경망 모델의 매개 변수는 기계가 읽을 수 있는, 단순히 긴 숫자로 이루어진 벡터입니다.</p>
</section>

<section>
  <h2 id="training">모델 트레이닝</h2>
  <p>이제 모델이 완성되었으니 트레이닝을 시작하십시오. 오른쪽 상단의 IntelliJ에서 녹색 화살표를 클릭하십시오. 위에서 설명한 코드가 실행됩니다.</p>
  <img src="/img/mlp_mnist_single_layer_example_training.png"><br><br>
  <p>하드웨어에 따라 완료하는 데 몇 분 정도 걸릴 수 있습니다.</p>
</section>

<section>
  <h2 id="evaluating">결과 평가</h2>
  <img src="/img/mlp_mnist_single_layer_example_results.png"><br><br>
  <p>
  <b>정확도</b> - 정확하게 식별 된 MNIST 이미지의 비율.<br>
  <b>정밀도</b> - True positives의 수를 True positives의 수와 False positives로 나눈 값.<br>
  <b>Recall</b> - True positives의 수를 True positives의 수와 False negatives의 수로 나눈 값.<br>
  <b>F1 Score</b> - <b>정확도</b>와 <b>리콜</b>의 가중 평균.<br>
  </p>
  <p><b>정확도</b>는 모델을 전반 측정합니다.</p>
  <p><b>정밀도, 리콜 및 F1</b>은 모델의 <b>관련성</b>을 측정합니다. 예를 들어, 더이상 치료를 받지 않을 것이기 때문에, 암이 재발하지 않을 것이라는 예측(즉, False negative)은 위험합니다. 이 때문에 전체 <b>정확도</b>가 낮더라도 False Negative (즉, 더 높은 정밀도, 리콜,  F1)는 피하는 모델을 선택하는 것이 좋습니다.</p>
</section>

<section>
  <h2 id="conclusion">결론</h2>
  <p>이제 다 되었습니다! 당신은 지금 컴퓨터 비전에 전혀 지식이 없던 신경망을97.1 %의 정확도를 달성할 수 있는 모델로 트레이닝 시켰습니다. 최신연구결과는 이것보다  훨씬 더 우수하며 하이퍼 파라미터를 추가로 조정한다면 모델을 더욱 향상시킬 수도 있습니다.</p>
  <p>다른 프레임워크와 비교했을 때, Deeplearning4j는 다음과 같은 장점을 가지고 있습니다.</p>
  <ul>
    <li>Spark, Hadoop, Kafka와 같은 주요 JVM 프레임워크와의 대거 통합이 가능</li>
    <li>분산 CPU 및 / 또는 GPU에서 실행을 위한 최적화</li>
    <li>Java 및 Scala 커뮤니티 서비스 지원</li>
    <li>엔터프라이즈의 배포를 위한 상업적 지원</li>
  </ul>
  <p>질문이 있으시면 Gitter(<a href="https://gitter.im/deeplearning4j/deeplearning4j" target="_blank">Gitter support chat room</a>)로 오십시오.</p>
  <ul class="categorized-view view-col-3">
    <li>
      <h5>기타 Deeplearning4j 튜토리얼</h5>
      <a href="http://deeplearning4j.org/neuralnet-overview">Introduction to Neural Networks</a>
      <a href="http://deeplearning4j.org/restrictedboltzmannmachine">Restricted Boltzmann Machines</a>
      <a href="http://deeplearning4j.org/eigenvector">Eigenvectors, Covariance, PCA and Entropy</a>
      <a href="http://deeplearning4j.org/lstm">LSTMs and Recurrent Networks</a>
      <a href="http://deeplearning4j.org/linear-regression">Neural Networks and Regression</a>
      <a href="http://deeplearning4j.org/convolutionalnets">Convolutional Networks</a>
    </li>

    <li>
      <h5>추천자료</h5>
      <a href="https://www.coursera.org/learn/machine-learning/home/week/1">Andrew Ng's Online Machine Learning Course</a>
      <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java">LeNet Example: MNIST With Convolutional Nets</a>
    </li>

  </ul>
</section>
