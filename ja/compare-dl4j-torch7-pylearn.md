---
title: 「ディープラーニングの比較シート：Deeplearning4j、Torch、Theano,TensorFlow、Caffe、Paddle、MxNet、Keras、CNTK
layout: default
---

# フレームワークの比較：Deeplearning4j、Torch、Theano, TensorFlow、Caffe、Paddle、MxNet、Keras、CNTK

Deeplearning4jは、その使用するAPI言語、意図すること、統合に関する点から他のフレームワークとは異なります。DL4Jは、JVMをベースとしており、業界向けの商用サポートが提供されている**分散型ディープラーニングフレームワーク**です。大量なデータに関連した問題を合理的な時間数で解決します。任意の数の[GPU](./gpu)や[CPU](./native)を使用してKafka、Hadoop、[Spark](./spark)と統合します。また、もし何かがうまく行かなかった場合には、[電話でお問い合せ](http://www.skymind.io/contact)いただけます。 

DL4Jは、AWS、Azure、Googleクラウドなどある1つのクラウドサービス向けに最適化されているというより、ポータブルでプラットフォームに関しては中立的です。速度については、複雑な画像処理タスクだとその[性能はCaffeと同レベル](https://github.com/deeplearning4j/dl4j-benchmark)であり、TensorflowやTorchよりも優れています。Deeplearning4jのベンチマーキングについての詳細情報は、こちらの[benchmarks page（ベンチマークのページ）](https://deeplearning4j.org/benchmark)をお読みください。このベンチマークは、JVMのヒープ領域、ガベージコレクションアルゴリズム、メモリ管理、DL4JのETLパイプラインを調節して性能を最適化するのに使用します。Deeplearning4jはJava、[Scala](https://github.com/deeplearning4j/scalnet)、[Kerasを使用しているPython API](./keras)を使用しています。

<p align="center">
<a href="quickstart" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">DEEPLEARNING4Jの使用を開始</a>
</p>

### 目次

Lua

* <a href="#intro">Torch及びPytorch</a>

Pythonのフレームワーク

* <a href="#intro">Theanoとエコシステム</a>
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
* <a href="#intro">ライセンス</a>

JVMについて

* <a href="#speed">速度</a>
* <a href="#java">DL4J：なぜJVMなのか？</a>
* <a href="#ecosystem">DL4J：エコシステム</a>
* <a href="#scala">DL4S：Scalaでのディープラーニング</a>
* <a href="#ml">機械学習のフレームワーク</a>
* <a href="#intro">その他の詳細</a>

## Lua

### <a name="torch">Torch及びPytorch</a>

[**Torch**](http://torch.ch/)は、Luaで書かれたAPIを持つ計算フレームワークで、機械学習アルゴリズムをサポートします。そのいくつかのバージョンはFacebookやTwitterなどの大手のIT企業でも使用されており、それぞれ社内チームを設けてディープラーニングプラットフォームのカスタマイズに従事しています。Luaはマルチパラダイムのスクリプト言語で、1990年代初めにブラジルで開発されました。 

Torch7はパワフルですが、Pythonを基盤とした学術的コミュニティーやJAVAを共通語とする企業ソフトエンジニアが[幅広くアクセスできることを目的として作成されたものではありませんでした](https://news.ycombinator.com/item?id=7929216)。Deeplearning4jがJavaで書かれた理由は、主に業界向けに適した言語を使用したいため、そして使いやすくするためです。ディープラーニングの普及を阻んでいるのは利便性に欠けることが要因ではないでしょうか。我々はHadoopやSparkなどのオープンソースの分散型ランタイムによって拡張性を自動化する必要があると考えています。そして、商業用にサポートされたオープンソースのフレームワークを使うことが、作業可能なツールを確保し、コミュニティーを築き上げるための適切な解決策だと考えています。

[Pytorch](https://github.com/pytorch/pytorch)として知られるTorchのPython APIは2017年1月にオープンソースになりました。PyTorchは様々な長さの入出力データを処理できる動的計算グラフを提供しています。これは例えば再帰型ニューラルネットワークを使用するときに役に立ちます。動的計算グラフをサポートしている他のフレームワークに、CMUのDyNet及びPFNのChainerがあります。 

長所と短所：

* (+) 簡単に組み合わせることのできるモジュールのピースが多くある。
* (+) 自分の層のタイプを簡単に入力し、GPU上で実行できる。
* (+) Lua (^.^)（ライブラリのコードのほとんどがLuaにあり、読みやすい。）
* (+) 前もってトレーニングされたモデルが多くある。
* (+) PyTorch
* (-) Lua
* (-) 通常、自分でトレーニングのコードを入力する（プラグ・アンド・プレイがあまりない）。
* (-) 商業用サポートがない。
* (-) ドキュメンテーションが不十分

## Pythonのフレームワーク

### <a name="theano">Theanoとエコシステム</a>

ディープラーニングの分野における研究者たちは、ディープラーニングのフレームワークの祖父であり、[Python](http://darkf.github.io/posts/problems-i-have-with-python.html)によって書かれた[**Theano**](http://deeplearning.net/software/theano/)を使っています。TheanoはNumpyのように多次元配列を処理するライブラリです。他のライブラリと使用し、データ探索に適しており、研究を目的としています。 

数多くのオープンソースのディープライブラリはTheanoの最上部に構築されています。これには[Keras](https://github.com/fchollet/keras)、[Lasagne](https://lasagne.readthedocs.org/en/latest/)、[Blocks](https://github.com/mila-udem/blocks)が含まれます。これらのライブラリは、常に直感的であるわけではないTheanoのインターフェイスの最上部に、使い方が簡単なAPIを配置させることを目的としています。（2016年3月以降、Theano関連のライブラリである[Pylearn2は使用不可となっているようです](https://github.com/lisa-lab/pylearn2))。

これとは対照的に、Deeplearning4jはディープラーニングを生産的環境へともたらし、JavaやScalaのようなJVM言語に解決策をもたらします。並列GPUやCPUにおいて、拡張性の高い方法でできるだけ多くのknobを自動化し、必要な場合はHadoop and [Spark](./spark.html)と統合することを目的としています。

長所と短所：

* (+) Python + Numpy
* (+) 計算グラフは見やすく抽象化されている。
* (+) 再帰型ニューラルネットワークは計算グラフに良く適合する。
* (-) 生のTheanoは幾分レベルが低い。
* (+) 高レベルのラッパー（Keras、Lasagne）により、手間が省ける。
* (-) エラーメッセージが役に立たないことがある。
* (-) 大きなモデルの場合、コンパイルの時間が長くなることがある。
* (-) Torchよりも「分厚い」
* (-) 事前トレーニングを受けたモデルにへのサポートが不十分
* (-) AWSでのバグが多い。
* (-) 単一GPU

### <a name="tensorflow">TensorFlow</a>

* GoogleはTheanoの代替としてTensorFlowを作成しました。これら二つのライブラリは実際には非常に類似しています。Ian Goodfellow氏などのTheanoの作成者のうち何人かは、OpenAI社へと移る前にGoogle社にてTensorflowも作成しました。 
* 現在のところ、**TensorFlow**は、いわゆる「インライン」の行列演算に対応しておらず、その上演算を実行するためには行列をコピーしなければなりません。非常に数の多い行列をコピーするのはあらゆる意味でコストがかかります。TensorFlowだと、最先端のディープラーニングツールの4倍の時間が掛かります。Googleはこの問題に取り組んでいる最中とのことです。 
* ほとんどのディープラーニングのフレームワークのように、TensorFlowはC/C++エンジンの上にあるPython APIによって書かれているため、より速く動作します。Java APIには実験的サポートが提供されていますが、現在それは安定したものとは考えられておらず、Java及びScalaのコミュニティーにとっては解決策であるとは考えられていません。 
* TensorFlowは、CNTKやMxNetなど[他のフレームワークと比べて動作は随分遅めです](https://arxiv.org/pdf/1608.07249v7.pdf)。 
* TensorFlowは、ディープラーニング以外のことができます。TensorFlowは、強化学習やその他のアルゴリズムに対応するツールを備えています。
* Googleも自認しているその目標とは、人員を増やし、研究者のコードを共有可能なものとし、ソフトウェアエンジニアのディープラーニングへのアプローチ方法を標準化し、TensorFlowが最適化されたGoogleのクラウドサービスの人気を高めることです。 
* TensorFlowは商用サポートを受けていません。Googleがオープンソースの企業ソフトウェアをサポートするビジネスを行うという可能性は低いでしょう。研究者に新しいツールを提供するものなのです。 
* Theanoのように、TensforFlowは計算グラフ（例：行列演算の一連であるz = sigmoid(x)、xやzは行列）を生成し、自動識別を行います。自動識別は重要です。これにより、新しいニューラルネットワークの配列で実験しているたびにバックプロパゲーションの新しいバリエーションをハンドコーディングする必要がなくなるからです。Googleのエコシステムでは、計算グラフはGoogle Brainが最も難しい仕事を処理するのにも使用しますが、Googleはこれらのツールをオープンソース化していません。TensorFlowはGoogleの社内デイープラーニング・ソリューションの半分を占めています。 
* ビジネスの観点から言うと、企業が考えるべきことは、これらのツールの提供をGoogleに求めたいかということです。 
* Tensorflowの演算すべてがNumpyと同じようには動作しないのでご注意ください。 

長所と短所：

* (+) Python + Numpy
* (+) Theanoと同じような計算グラフの抽象化
* (+) Theanoよりもコンパイル時間が短い。
* (+) TensorBoardを使用して視覚化する。
* (+) データとモデルの並列処理
* (-) 他のフレームワークより遅い。
* (-) Torchよりも「ぶ厚い」。魔法のようなことがもっとある。
* (-) 事前トレーニングされたモデルがあまりない。
* (-) 計算グラフは単なるPythonなため、遅い。
* (-) 商用サポート無し
* (-) 新しいトレーニングのバッチがあるごとに読み込みをするためにPythonにドロップアウトする。
* (-) ツールにはなりにくい。
* (-) 大規模なソフトウェアプロジェクトで動的型付けにエラーが発生しやくすい。

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/)は有名で、広く一般に使用されているマシンビジョンのライブラリで、Matlabのの実装した素早い畳み込みネットワークをC及びC++で移植したものです（速度とテクニカル面の負担のトレードオフを考慮したい方は、あるチップから別のチップにC++を移植することについてのSteve Yegge氏の見解を[こちら](https://sites.google.com/site/steveyegge2/google-at-delphi)でお読みいただけます）。Caffeは、テキスト、音声、時系列データなど他のディープラーニングのアプリケーションを対象としたものではありません。これまでご紹介してまいりました他のフレームワークと同様に、CaffeはそのAPIが目的でPythonを選択しました。 

Deeplearning4jとCaffeはどちらも畳み込みネットワークを使用して、最先端の技術で画像分類を行います。Caffeとは対照的に、Deeplearning4jはGPUの並列処理*サポート*を提供しているため、任意数のチップに使用できるだけでなく、数多くの特徴に対応できるため、ディープラーニングがよりスムースに複数のGPUクラスタを使って並列処理できます。論文でよく引用されていますが、CaffeはそのModel Zooサイトがホストする事前トレーニングされたモデルのソースとして主に使用されます。

長所と短所：

* (+) フィードフォワード（順伝播型）ネットワークと画像処理に適している。
* (+) 既存のネットワークの微調整に適している。
* (+) コードを書かずにモデルをトレーニングする。
* (+) Pythonのインターフェースが非常に有益である。
* (-) 新しいGPU層にC++/CUDAを書く必要がある。
* (-) 再帰型ニューラルネットワークには適していない。
* (-) 大規模なネットワークには（GoogLeNet, ResNet）使いにくい。
* (-) 拡張性がない。
* (-) 商業用サポートがない。
* (-) 使用されなくなりつつある可能性が高い。開発に時間が掛かる。

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK)とは、マイクロソフトによるオープンソースのディープラーニングフレームワークで、CNTKとは「Computational Network Toolkit（計算ネットワークのツールキット）」の略語です。そのライブラリにはフィードフォワードディープニューラルネットワーク、畳み込みネットワーク、再帰型ニューラルネットワークネットが含まれています。そして、C++コードでPython APIを提供しています。CNTKは[permiLICENSEssive license（許容的ライセンス）](https://github.com/Microsoft/CNTK/blob/master/.md)を取得していると誤解されがちですが、ASF 2.0、BSD、MIT.など従来のどのライセンスも採用していません。許容的ライセンスはCNTKの1ビットSGD（確率的勾配降下法）といって、分散型トレーニングが簡単になる方法には適用されません。商業用には適用されないライセンスだからです。 

### <a name="chainer">Chainer</a>

Chainerとはオープンソースのニューラルネットワークフレームワークで、Python APIを備えています。中心となる開発チームは[Preferred Networks](https://www.crunchbase.com/organization/preferred-networks#/entity)という東京を拠点とする機械学習のスタートアップで活動しており、多くの東京大学のエンジニアが携わっています。CMUのDyNetやFacebookのPyTorchが登場するまではChainerが動的計算グラフの主要なニューラルネットワークフレームワークであり、様々な長さの入力が可能なネットワークであったため、自然言語処理作業には人気の高い機能です。独自の[ベンチマーク](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html)を使い、Chainerは他のPython向けフレームワークよりも素早く、MxNetとCNTKを含めた最も遅いテストグループであるTensorFlowを備えています。 

### <a name="dsstne">DSSTNE</a>

Amazonの[DSSTNE](https://github.com/amznlabs/amazon-dsstne)（Deep Scalable Sparse Tensor Network Engine）は機械学習及びディープラーニングのモデル構築用のライブラリです。DSSTNEは、TensorflowやCNTKに次いで、近々リリースが予定されている多くのオープンソース・ディープラーニング・ライブラリのうちの一つです。AmazonはすでにAWS（Amazon Web Services）によってMxNetをサポートしているため、その将来は明確ではありません。DSSTNEはそのほとんどがC++コードで書かれており、他のライブラリほど人気は集めていませんが、作業速度は素早いようです。 

* (+) スパースエンコーデイングに対応
* (-) Amazonは[最高結果を得るために必要なすべての情報やそのサンプル](https://github.com/amznlabs/amazon-dsstne/issues/24)をシェアしていない可能性がある。
* (-) Amazonは別のフレームワークを選択してAWSに使用している。

### <a name="dynet">DyNet</a>

[DyNet](https://github.com/clab/dynet)（[Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980)）はカーネギーメロン大学が開発し、元々CNNと呼ばれていました。その注目すべき機能は、動的計算グラフであり、様々な長さの入力が可能なため自然言語処理作業に適しています。PyTorch及びChainerも同じ機能を備えています。 

* (+) 動的計算グラフ
* (-) ユーザコミュニティが小規模

### <a name="keras">Keras</a>

[Keras](keras.io)とは、Theano及びTensorFlowの最上部で使用するディープラーニングライブラリで、Torchを模範とした直感的なAPIを提供しています。現在利用可能なPython APIの中でも最も優れたものかもしれません。Deeplearning4jの[Python API](./keras)はKerasを使用しており、[Kerasから、そしてKerasを通じてTheanoやTensorFlowからモデルをインポートしています](./model-import-keras)。作成者はGoogleのソフトウェアエンジニアの[Francois Chollet氏](https://twitter.com/fchollet)です。 

* (+) Torchを模範とした直感的なAPI
* (+) Theano、TensorFlow、Deeplearning4jのバックエンドに対応（CNTKバックエンドも今後追加される予定）
* (+) フレームワークが急速な成長を続けている。
* (+) ニューラルネットワークの標準的なPython APIとなる可能性が高い。

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet)はAPIのある機械学習フレームワークで、言語はR、Python、Juliaなど[Amazonのウェブサービスが採用する](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)言語を使うことができます。Apple社も2016年のGraphlab/Dato/Turi社買収後にApple製品の一部にこれを使用するとも言われています。MxNetは高速で柔軟性の高いライブラリで、Pedro Domingos氏及びワシントン大学の研究者チームがその開発に従事しています。MxNetとDeeplearning4jのいくつかの特徴の[比較分析](https://deeplearning4j.org/mxnet)をこちらでお読みいただけます。 

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle)は、[Baidu氏が作成し、サポートする](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html)ディープラーニングフレームワークです。その正式名称はPArallel Distributed Deep LEarningです。Paddleはリリースが予定されているフレームワークのうち、最も主要なものです。その他のフレームワークと同様、Python APIを提供しています。 

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL)は新しいディープラーニングフレームワークで、Apache Sparkに主に重点を置いており、Intelチップのみで使用可能です。 

### <a name="licensing">ライセンス</a>

これらのオープンソースはそのライセンスよっても分類されます。Theano、Torch、CaffeはBSDライセンスを採用していますが、このライセンスは特許や特許に関連した紛争は扱っていません。Deeplearning4j及びND4Jは**[Apache 2.0 ライセンス](http://en.swpat.org/wiki/Patent_clauses_in_software_licences#Apache_License_2.0)**の下で配布されていますが、このライセンスは特許許可と報復的訴訟の条項の両方を扱っています。つまり、Apache 2.0のライセンスを持つコードを使用して特許や派生物を作成してもいいのですが、元のコード（この場合はDL4J）に関する特許請求の範囲について誰かを訴えた場合、直ちにすべての特許請求の範囲を失うということです。（要するに、訴訟で自分を弁護するためのリソースは与えられていますが、他人を攻撃するのは控えた方がいいということになります。）BSDライセンスは一般的にはこの問題を扱っていません。 

## JVMについて

### <a name="speed">速度</a>

ND4Jで実行されるDeeplearning4jの基盤である線形代数計算は、[Numpyよりも最低2倍の速度](http://nd4j.org/benchmarking)で大規模な行列の積を計算するということが証明されています。このため、NASAのJet Propulsion Laboratory（ジェット推進研究所）のチームは弊社の技術を採用しています。また、Deeplearning4jはx86やCUDA CのGPUを含む様々なチップ実行できるように最適化されています。

Torch7もDL4Jも並列処理を行いますが、DL4Jの**並列処理は自動**で実行されます。つまり、ノードや接続の設定を自動化し、ライブラリを通ることなく、[Spark](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark)、[Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn)に並列ネットワークを作成したり、[Akka及びAWS](http://deeplearning4j.org/scaleout.html)を使用できるのです。Deeplearning4jは特定の問題解決を素早く行うのに最適です。 

Deeplearning4jの機能の全リストについては弊社の[features（機能）のページ](./features.html)をご覧ください。

### <a name="java">なぜJVMなのか？</a>

我々がよく受ける質問に、ディープラーニングのコミュニティーの多くがPythonを重視する中、なぜJVM向けのオープンソースのディープラーニング・プロジェクトを導入するのか、というものがあります。PythonはJavaとは異なり、構文の要素がたくさんあるため、explicitクラスを作成せずに行列を追加することができます。同様にして、PythonにはTheanoやNumpyなどネイティブ拡張があり、広範囲に渡って科学計算が行える環境があります。

しかし、JVMやその主要言語（JavaやScalaなど）にはいくつかの長所があります。 

まず、最大手企業や大規模な政府機関で主に使用されているのはJavaやJVMベースのシステムだということです。これらの組織は巨大な投資を行っているため、JVMベースのAIを利用することが可能です。今でもJavaは事業に最も幅広く使用されているのです。Javaは機械学習の問題に対処するのに最も役に立つHadoop、ElasticSearch、Hive、Lucene、Pigに使用される言語なのです。SparkやKafkaはScalaで書かれていますが、これもJVMの言語です。つまり、現実問題を解決している多くのプログラマーはディープラーニングの恩恵を受けることが可能なのに言語の障壁によって引き離されているのです。我々はディープラーニングをもっと多くの新しい利用者がすぐに使用できるように進化させていきたいと考えています。Javaは1千万人もの開発者がいる世界最大規模のプログラミング言語です。 

そして、JavaとScalaはPythonより本質的に素早いというのも理由です。Pythonで書かれたものはどれでもCythonを使用していることを差し引いても速度は遅めになります。確かに最も計算的に高価な演算は、C言語やC++言語で書かれています。（演算についてとなると、より高レベルな機械学習処理に関連した文字列やその他の作業についても考慮しなければなりません。）元々Pythonで書かれたほとんどのディープラーニングのプロジェクトは、生産的にするためには再度書き直さなければなりません。Deeplearning4jは、Javaから事前にコンパイルされたネイティブのC++を呼び出すために[JavaCPP](https://github.com/bytedeco/javacpp)を使用し、トレーニングの速度を著しく向上させています。多くのPythonのプログラマーはScalaでディープラーニングを行おうとしますが、これは他の開発者たちと共有したコードベースで作業するときに静的型付けや機能的プログラミングを行いたいからです。 

また、Javaに強固な科学計算ライブラリはないという問題はそれらを書きさえすれば解決するのです。これは既に[ND4J](http://nd4j.org)で行いました。ND4Jは分散型のGPUやGPUで実行されます。そしてJavaやScala API経由で相互作用させることができます。

最後に、Javaは本来はLinuxのサーバー、Windows、OSXデスクトップ、アンドロイドの電話、モノのインターネットの低メモリーセンサーなど埋め込まれたJava経由で異なるプラットフォーム間で動作する安全が確保されたネットワーク言語です。TorchやPylearn2はC++経由で最適化しますが、この最適化や維持には困難が伴います。それに比べてJavaは「一度書くと、どこでも実行」可能な言語で、ディープラーニングを数多くのプラットフォームで使用する必要のある企業には適しています。 

### <a name="ecosystem">エコシステム</a>

Javaの人気はそのエコシステムによりますます高まりました。[Hadoop](https://hadoop.apache.org/)はJavaに実装されています。[Spark](https://spark.apache.org/)はHadoopのYarn実行時間内に実行されます。[Akka](https://www.typesafe.com/community/core-projects/akka)などのライブラリによってDeeplearning4jの分散型システムの構築が可能になりました。まとめると、Javaは大体どのアプリケ―ションにも高度に検証されたインフラストラクチャーで、Javaで書かれたディープラーニングネットワークはデータの近くに位置することができ、プログラマーの生活がより簡単になるのです。Deeplearning4jはYARNアプリとして実行し、提供することができます。

また、JavaはScala、Clojure、Python、Rubyなど他の人気のある言語からネイティブで使用することができます。Javaを選ぶことにより、我々はできるだけ最も少ない主要なプログラミングのコミュニティーを除外しました。 

JavaはCやC++ほど素早くはありませんが、多くの人が思い込んでいるよりも速度は速いため、我々はGPUであろうとCPUであろうとノードを少し追加すると加速できるよう、分散システムを構築しました。つまり、速度を向上させたければ、より多くのボックスをスローすればいいのです。 

最後に我々はJavaにDL4J用のND-Arrayを含めて、Numpyの基本的アプリケーションを構築しています。我々はJavaの欠点の多くは素早く解決できるものであり、長所の多くもすぐにはなくならないはずだと見ています。 

### <a name="scala">Scala</a>

我々はDeeplearning4jとND4Jを構築するにあたって[Scala](./scala)に特に注目しましました。データサイエンスにおいてScalaが優勢な言語となる可能性があると見ているからです。数値計算を書くこと、ベクトル化、[Scala API](http://nd4j.org/scala.html)の実装されたJVM用のディープラーニングライブラリによってコミュニティーはその目標に向かうことができるのです。 

DL4Jとその他のフレームワークの違いの理解を深めるには、[弊社のサービスを試してみる](./quickstart)のがお勧めです。

### <a name="ml">機械学習のフレームワーク</a>

先に挙げたようなディープラーニングのフレームワークは一般的な機械学習フレームワークよりも専門化されたものです。現在、専門化は機械学習フレームは数多くあります。ここでは主なものだけを挙げましょう。

* [sci-kit learn](http://scikit-learn.org/stable/) - Python向けのデフォルトの機械学習フレームワーク 
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html) - 最も重要なApacheの機械学習フレームワーク。Mahoutが分類、クラスタリング、リコメンデーションを行います。
* [SystemML](https://sparktc.github.io/systemml/quick-start-guide.html) - IBMの機械学習フレームワークで、記述統計学、分類、クラスタリング、回帰、行列の因数分解、生存率解析を行う。サポートベクターマシンを含みます。 
* [Microsoft DMTK](http://www.dmtk.io/) - Microsoftの分散型機械学習ツールキット分散型の語の埋め込みとLDA（線形判別分析） 

### <a name="tutorial">Deeplearning4jのチュートリアル</a>

* [ディープニューラルネットワークについて](./neuralnet-overview)
* [畳込みネットワークのチュートリアル](./convolutionalnets)
* [LSTMと回帰ネットワークのチュートリアル](./lstm)
* [回帰ネットワークをDL4Jに使用しましょう](./usingrnns)
* [ディープ・ビリーフ・ネットワークとMNIST](./deepbeliefnetwork)
* [DataVecを使った Data Pipelineのカスタマイズ](./image-data-pipeline)
* [制限付きボルツマン・マシン](./restrictedboltzmannmachine)
* [固有ベクトル、PCA（主成分分析）、エントリピー](./eigenvector.html)
* [ディープラーニング用語集](./glossary.html)
* [Word2vec、Doc2vec、GloVe](./word2vec)
