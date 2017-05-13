---
title:初心者のためのMNIST
layout: default
---

<header>
  <h1>初心者のためのMNIST</h1>
  < p >このチュートリアルでは、MNISTデータセットを分類します。< / p >
  <ol class="toc">
    <li><a href="#introduction">はじめに</a></li>
    <li><a href="#mnist-dataset">MNISTのデータセット</a></li>
    <li><a href="#configuring">MNISTのサンプルの設定</a></li>
    <li><a href="#building">ニューラルネットワークの構築</a></li>
    <li><a href="#training">モデルのトレーニング</a></li>
    <li><a href="#evaluating">結果の評価</a></li>
    <li><a href="#conclusion">最後に</a></li>
  </ol>
</header>
  <p>このページを終了するのに要する予測時間</p>
<section>
  <h2 id="introduction">はじめに</h2>
  <img src="/img/mnist_render.png"><br><br>
  <p>MNISTとは、手書き数字画像のデータベースです。各画像は整数によってラベル付けされています。機械学習アルゴリズムのの性能の便意マーク使用されます。ディープラーニングをMNISTに適用すると、非常によい性能を発揮し、正答率は99.7%となっています。</p>
  < p >MNISTを使用して、ニューラルネットワークが画像から数字を推測できるよう、トレーニングを行います。まず最初にDeeplearning4jをインストールします。</p>
  <a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ?equickstart', 'click');">DEEPLEARNING4Jの使用を開始</a>
</section>

<section>
  <h2 id="mnist-dataset">MNISTのデータセット</h2>
  <p>MNISTのデータセットには、60,000のサンプルを含んだ<b>トレーニングセット</b>一部と10,000のサンプルを含んだ<b>テストセット</b>で構成されています。トレーニングセットは、正確なラベルである整数を推測できるよう訓練するために使用されており、テストセットはトレーニングされたネットワークがどれだけの正答率で推測できるかをチェックするために使用されます。</p>
  <p>機械学習の世界では、これは<a href="https://ja.wikipedia.org/wiki/%E6%95%99%E5%B8%AB%E3%81%82%E3%82%8A%E5%AD%A6%E7%BF%92" target="_blank">教師あり学習（supervised learning）</a>と呼ばれています。推測する画像に正しい答えがあるからです。したがって、推測が間違っていた場合、トレーニングセットは、教師、先生という役割を担うことができます。</p>
</section>

<section>
  <h2 id="configuring">MNISTの設定例</h2>
  < p >我々はMNISTのチュートリアルをMavenでパッケージ化したので、コードを記述する必要はありません。IntelliJを開いて開始してください。（IntelliJをダウンロードするには、弊社の<a href="quickstart">Quickstart?c</a>をお読みください。</p>
  <p><code>dl4j-examples</code>とラベルされたフォルダを開いてください。ディレクトリの<kbd>src</kbd> → <kbd>main</kbd> → <kbd>java</kbd> → <kbd>feedforward</kbd> → <kbd>mnist</kbd>へと進み、ファイルの<code>MLPMnistSingleLayerExample.java</code>を開いてください。</p>
  <p><img src="/img/mlp_mnist_single_layer_example_setup.png"></p>
  <p>このファイルでは、ニューラルネットワークを設定し、モデルのトレーニングを行い、結果を評価します。このコードをチュートリアルと一緒に見ると便利です。</p>
  <h3>変数の設定</h3>
  <pre><code class="language-java">
    final int numRows = 28; // 行列の行数
    final int numColumns = 28; // 行列の列数
    int outputNum = 10; // 可能な結果の数（例：ラベルの0から9）
    int batchSize = 128; // 各ステップでいくつのサンプルを使うか
    int rngSeed = 123; // この乱数発生器はシードを適用して同じ初期の重みがトレーニングに使用されていることを確保します。なぜこれが重要なのかについては後ほど説明します。
    int numEpochs = 15; // エポックとは対象とするデータセットが完全に通過した回数
  </code></pre>
  <p>弊社の例では、MNISTの各画像は28x28画素であり、 つまり入力データは28 <b>numRows</b> x 28 <b>numColumns</b>の行列（行列はディープラーニングのデータ構造の基盤）であるということになります。また、MNISTには可能な結果が10あります（0から9までの番号が付けられたラベル）。これは、弊社の<b>outputNum</b>に当たります。</p>
  <p><b>batchSize</b>と<b>numEpochs</b>は経験に基づいて選択します。これは実験を重ねていくにつれて分かってきます。高速トレーニングのバッチサイズが大きいとトレーニングが速くなり、エポックやデータセット内の通過が多いと正答率が向上します。</p >
  <p>しかし、ある一定の数を超えるエポックに対するリターンは減少するため、正答率とトレーニングの速度との間にはトレードオフがあります。一般的には、最適な値を突き止めるまで実験を続ける必要があります。弊社はこの例において妥当なデフォルト値を設定しました。</p>
  <h3>MNISTのデータ</h3>
  <pre><code class="language-java">
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
  </code></pre>
  <p><code>DataSetIterator</code>と呼ばれるクラスは、MNISTのデータセットをフェッチする（取り出す）ために使用されます。我々はある1つのデータセット<code>mnistTrain</code>を作成して<b>モデルのトーレーニング</b>を行い、もう一つのデータセット<code>mnistTest</code>を作成してトレーニング後のモデルの<b>正答率を評価</b>します。ところで、このモデルは、ニューラルネットワークのパラメータを参照します。これらのパラメータは、入力データの信号を処理する係数であり、ネットワークが各画像の正しいラベルを推測できるようになり、最終的に正確なモデルとなるまでこれらの係数は調整されます。</p>
</section>

<section>
  <h2 id="building">Building Our Neural Network</h2>
  <p>あるフィードフォワード（順伝播型）ネットワークを<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank">Xavier Glorot and Yoshua Bengioによる論文</a>に基づいて構築しましょう。ここでは、隠れ層が一つだけの簡単な例で始めましょう。しかし、経験から言うと、ネットワークが深ければ深いほど（つまり層が多ければ多いほど）、より複雑で微妙な部分を取り込み、正確な結果を出すことができます。</p>
  <img src="/img/onelayer.png"><br><br>
  <p>この図をよく覚えておいてください。というのは、これから我々はこの1層のニューラルネットワークを構築するからです。</p>
  <h3>ハイパーパラメータの設定</h3>
  <p>Deeplearning4jで構築するいかなるニューラルネットワークでも、基盤は<a href="http://deeplearning4j.org/neuralnet-configuration.html" target="_blank">NeuralNetConfigurationクラス</a>です。ここでアーキテクチャの数量とアルゴリズムの学習方法を定義するハイパーパラメータを設定します。感覚的な例だと、各パラメタ―はある料理に使う食材のうちの一つのようなもので、これによって料理の成功、失敗が大きく左右されるようなものです。幸い、正しい結果が生み出されなければ、ハイパーパラメータを調整することができます。</p>
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
  <p>このパラメータはある特定のランダムに生成された重量の初期化を使用します。ある例を何回も実行し、毎回開始時に新しい重みを生成すると、ネットの結果（F1スコアと正答率）にかなりの違いがもたらされるかもしれません。というのは、初期の重みが異なるとルゴリズムでエラースケープの極小値が異なってしまうかもしれないからです。重みを同じランダムなものに保っておくと、他の条件を平等に保ったまま、他のハイパーパラメタ―を調整する効果をもっと明確に隔てることができます。</p>
  <h5>.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)</h4>
  <p>Stochastic gradient descent (SGD)は、コスト関数を最適化するための一般的な方法です。エラーを最小限の抑えるSGDや他の最適化アルゴリズムについて知るには、<a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng?fs Machine Learning course</a>、弊社の<a href="http://deeplearning4j.org/glossary#stochasticgradientdescent" target="_blank">グロッサリー</a>にあるSGDの定義をご参照ください。</p>
  <h5>.iterations(1)</h5>
  <p>一回のニューラルネットワークのイテレーションは、学習の一段階に当たります。つまり、モデルの重み更新の一回分に当たるのです。ネットワークはデータを目の当たりにし、そのデータについて推測し、その推測がどのくらい間違っていたかに基づいて自身のパラメータを修正します。イテレーションを多く行えば行うほど、ニューラルネットワークはより多くの段階を踏み、学習することができ、エラーを最小限に抑えることができるのです。</p>
  <h5>.learningRate(0.006)</h5>
  <p>このコマンドは、学習率としてイテレーション一回における重み調整のサイズ、つまりステップサイズを指定します。学習率が高いとネットはerrorscapeを素早く巡回しますが、のトラバースになり、すぐに errorscape 、も最小エラーのポイントをオーバーシュートしやすい傾向があります。学習速度が低いと、最小値を見つける可能性は高まりますが、非常に遅く行われます。小さいステップで重みを調整するためです。</p>
  <h5>.updater(Updater.NESTEROVS).momentum(0.9)</h5>
  <p>運動量（momentum）は最適なポイントに最適化アルゴリズムがどれだけ素早く収束するかを決定する要素のうち一つです。運動量は、重みが調整される方向に影響するため、コーディングの世界では、一種の重みの<code>アップデーター</code>と見なします。</p>
  <h5>.regularization(true).l2(1e-4)</h5>
  <p>正規化とは、<b>過剰適合（overfitting）</b>を回避するためのテクニックです。過剰適合とは、あるモデルがトレーニングのデータには非常によく適合しても、実際に使用された時に過去に接したことのないデータに出くわすやいなや非常にパフォーマンスが悪くなることを言います。</p>
  <p>弊社では、L2正規化を使用することにより、個々の重みが全体の結果に大きな影響を及ぼさないよう回避しています。</p>
  <h5>.list()</h5>
  <p>ネットワーク内の層数を指定します。この関数は、自分の設定をn回複製し、層の設定を構築します。</p>
  <p>上記の説明で分かりにくいことがあれば、先にも触れた<a href="https://www.coursera.org/learn/machine-learning" target="_blank">Andrew Ng?fs Machine Learning course</a>をご参照になることをお勧めします。</p>
  <h3>層の構築</h3>
  <p>ここでは各ハイパーパラメタ―の背景（活性化、weightInit）については取り上げませんが、それらの役割について簡単に触れておきましょう。ただし、これらがなぜ重要なのかを知りたい方は、<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf" target="_blank">Xavier Glorot and Yoshua Bengioによる論文</a>をお読みください。
  <img src="/img/onelayer_labeled.png"><br>
  <pre><code class="language-java">
    .layer(0, new DenseLayer.Builder()
            .nIn(numRows * numColumns) //入力データポイントの数
            .nOut(1000) // 出力データポイントの数
            .activation("relu") // 活性化の関数
            .weightInit(WeightInit.XAVIER) // 重みの初期化
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
  <p>ところで隠れ層とは一体何なのでしょうか？</p>
  <p>隠れ層の各ノード（図式の丸）はMNISTデータセットでは、手書きの数字の特徴を表しています。例えば、数字の「6」を見ているとすれば、あるノードは丸い角を表し、別のノードは交差した曲線を表したもの、となっています。これらの特徴はモデルの係数の重要性によって重みを付与され、この手書きの数字が実際に「6」なのか推測するために各隠れ層で再度組み合わせられます。ノードの層が多ければ多いほど、推測の向上に必要な複雑さやニュアンスを取り込むことができます。</p>
  <p>「隠れ」層と見なしてもいいわけは、入力データがネット内に入り、決断が出てくるのを見ることができても、どのようにして、またなぜニューラルネットワークがデータを処理しているかを解読することは人間には不可能だからです。ニューラルネットワークのパラメーターは、機械のみが読み取ることのできる長い数字のベクトルに過ぎないのです。</p>
</section>

<section>
  <h2 id="training">モデルのトレーニング</h2>
  <p>モデルが構築できたら、トレーニング を開始しましょう。IntelliJの右上で、緑色の矢印をクリックします。この操作により、上記のコードが実行されます。</p>
  <img src="/img/mlp_mnist_single_layer_example_training.png"><br><br>
  <p>これはハードウェアによっては、完了に数分掛かることがあります。</p>
</section>

<section>
  <h2 id="evaluating">結果を評価</h2>
  <img src="/img/mlp_mnist_single_layer_example_results.png"><br><br>
  <p>
  <b>Accuracy（正答率）</b> - モデルが正しく識別したMNIST画像の割合<br>
  <b>Precision（適合率）</b> - 真陽性の数を真陽性と偽陽性の数で割った値<br>
  <b>Recall（再現率）</b> - 真陽性の数を真陽性の数と偽陰性の数で割った値<br>
  <b>F1値</b> - <b>適合率</b>と<b>再現率</b>の加重平均<br>
  </p>
  <p><b>正答率</b>はモデル全体を測定します。</p>
  <p><b>適合率、再現率、F1</b>はモデルの<b>relevance</b>を測定します。例えば、ある人がさらなる治療を求めないため、癌は再発しないだろうと（偽陰性）と推測するのは危険なことでしょう。このため、全体的に<b>accuracy</b>が低くめでも偽陰性（つまりprecision、recall、F1が高め）を回避するモデルを選択するのが賢明でしょう。</p>
</section>

<section>
  <h2 id="conclusion">最後に</h2>
  <p>これで手に入りました！コンピュータビジョンが0ドメイン知識であるニューラルネットワークをトレーニングし、正答率（accuray）の97.1%を達成しました。最先端のパフォーマンスは、これよりさらに優れており、ハイパーパラメタ―をさらに調整してモデルを改善させることができます。</p>
  <p>その他のフレームワークと比較すると、Deeplearning4jは以下の点で優れています。</p>
  <ul>
    <li>規模を広げて、Spark、Hadoop、Kafkaなどの主要なJVMフレームワークと統合させることができる。</li>
    <li>分散CPUと/または分散GPUで実行に最適化されている。</li>
    <li>JavaやScalaのコミュニティに貢献している。</li>
    <li>導入された企業様への商業的サポート</li>
  </ul>
  <p>ご質問のある方は弊社のオンライン<a href="https://gitter.im/deeplearning4j/deeplearning4j" target="_blank">Gitter サポートチャットルーム</a>にてご連絡ください。</p>
  <ul class="categorized-view view-col-3">
    <li>
      <h5>その他のDeeplearning4jのチュートリアル</h5>
      <a href="http://deeplearning4j.org/neuralnet-overview">Introduction to Neural Networks</a>
      <a href="http://deeplearning4j.org/restrictedboltzmannmachine">Restricted Boltzmann Machines</a>
      <a href="http://deeplearning4j.org/eigenvector">Eigenvectors, Covariance, PCA and Entropy</a>
      <a href="http://deeplearning4j.org/lstm">LSTMs and Recurrent Networks</a>
      <a href="http://deeplearning4j.org/linear-regression">Neural Networks and Regression</a>
      <a href="http://deeplearning4j.org/convolutionalnets">Convolutional Networks</a>
    </li>

    <li>
      <h5>おすすめのリソース</h5>
      <a href="https://www.coursera.org/learn/machine-learning/home/week/1">Andrew Ng's Online Machine Learning Course</a>
      <a href="https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java">LeNet Example:MNIST With Convolutional Nets</a>
    </li>

  </ul>
</section>
