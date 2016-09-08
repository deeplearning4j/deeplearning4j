---
title: "Sparkを使ったDeeplearning4j"
layout: ja-default
---

# Sparkを使ったDeeplearning4j

ディープラーニングは、多くの計算処理を必要とするため、かなり大規模なデータセットを使用する場合は、そのスピードの速さが重要になってきます。より速度を早めるには、高速なハードウェア（通常、GPU）、最適化されたコード、ある種の並列処理によって対処することができます。 

データ並列処理は大規模なデータセットを複数のサブセットに分け、それらを別々のニューラルネットワーク、コアに提供します。Deeplearning4jは、この作業をSparkを使って行います。複数のモデルを並行してトレーニングし、中央のモデルに産出するパラメータの[繰り返し平均化処理](./iterativereduce.html)を行います。(モデルの並列処理は、[Jeff Dean et alによって論じられていますが](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)、これにより、大規模データセットの平均を出さなくてもモデルは別々のパッチを処理することができます。）

**目次**

* [概要](#overview)
* [DL4JのSpark上での分散ネットワークのトレーニング](#how)
* [必要最低限のコンポーネントを使った例](#minimal)
* [TrainingMasterの設定](#configuring)
* [Sparkでトレーニングをする場合の依存関係](#dependencies)
* [Sparkを使った例のリポジトリ](#examples)
* [YARN上でSparkもメモリを設定](#memoryyarn)
    * [Deeplearning4j (及びND4J）のメモリ管理方法](#memory1)
    * [YARNのメモリ管理](#memory2)
    * [Deeplearning4jのYARN上でのSparkのトレーニング用にメモリ設定](#memory3)
* [KryoシリアライゼーションのDeeplearning4jでの使用](#kryo)
* [KryoシリアライゼーションのDeeplearning4jでの使用](#mklemr)

## <a name="overview">概要</a>

Deeplearning4jは、Sparkのクラスタ上のニューラルネットワークのトレーニングを行うことにより、ネットワークのトレーニングの加速化を実現しています。

DL4JのMultiLayerNetworkクラスやComputatiDL4JonGraphクラスのように、DL4JはニューラルネットワークのトレーニングをSpark上で行うために2つのクラスを設定しています。

- SparkDl4jMultiLayerと言うMultiLayerNetworkに使用するラッパー
- SparkComputationGraphと言うaround ComputationGraphに使用するラッパー

これら2つのクラスは、標準的な単一マシンクラスのラッパーなので、ネットワークの設定のプロセス（すなわち、MultiLayerConfigurationやComputationGraphConfigurationの作成）は標準的な分散トレーニングと全く同じです。とはいえ、Sparkを使った分散トレーニングは、データの読み込み方法とトレーニングの設定方法という2つの点でローカルトレーニングとは異なります（追加のクラスタ特有の設定が必要）。

Sparkのクラスター上で行う一般的なトレーニングの流れ（Spark-submitを使います）は以下の通りです。

1.トレーニングクラスを作成します。この際、通常は以下を行うためのコードが必要です。
    * ネットワーク設定の指定（MultiLayerConfigurationまたはComputationGraphConfiguration）。単一機械でのトレーニングと同様です。
    * TrainingMasterインスタンスの作成。これにより分散トレーニングが実際にどのように行われるかが指定されます（この詳細は後述します）。
    * ネットワーク設定及びTrainingMasterオブジェクトを使用してSparkDl4jMultiLayerまたはSparkComputationGraphのインスタンスを作成
    * トレーニングするデータの読み込み。データを読み込む方法には数多くあり、トレードオフも異なります。詳細は、別の機会にご紹介する予定です。
    * 適切な```fit（当てはめ）```メソッドをSparkDl4jMultiLayerまたはSparkComputationGraphインスタンスに呼び出し
    * トレーニングしたネットワークの保存または使用（トレーニングしたMultiLayerNetworkまたはComputationGraphのインスタンス）
2.Spark submit用のjarファイルをパッケージ化します。
    * Mavenを使用している場合は、「mvn package -DskipTests」を実行させるのも一つのアプローチ方法です。
3.自分のクラスタの適切な起動設定（launch configuration）でSpark submitを呼び出します。


**注意**単一の機械によるトレーニングに、Spark localを使用することは*可能*ですが、あまりおすすめはできません（Sparkの同期化とシリアライゼーションのオーバーヘッドが発生するため）。その代わりに、以下を考えてみてください。

* 単一のCPU/GPUシステムの場合、標準的なMultiLayerNetworkまたはComputationGraphのトレーニングを使用する。
* 複数のCPU/GPUシステムには、[ParallelWrapper](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/parallelism/ParallelWrapper.java)を使用する。これは、機能的にはSparkをローカルモードで実行するのに等しいですが、オーバーヘッドが軽減します（このためトレーニングのパフォーマンスが向上します）。 

## <a name="how">DL4JのSpark上での分散ネットワークのトレーニング</a>

DL4Jの現在のバージョンでは、ネットワークのトレーニングにパラメータの平均化処理を適用しています。今後のバージョンには、さらにその他の分散ネットワークのトレーニングアプローチが含まれる可能性もあります。


パラメータ平均化処理によりネットワークのトレーニングを行うのは概念的には非常にシンプルです。

1.Master（Sparkのドライバ）は、最初のネットワーク設定とパラメータがあれば開始します。
2.データは、TrainingMasterの設定に基づいて数多くのサブセットへと分割します。
3.分割されたデータに繰り返し処理を行います。トレーニングデータの各分割に以下の作業を行います。
    * 設定、パラメータ（該当する場合は、momentum/rmsprop/adagradのNetwork Updaterの状態）をMasterから各Workerに分配する。
    * 各workerを分割の割当分に当てはめる。
    * パラメータ（そして該当する場合はNetwork Updaterの状態）の平均化を行い、平均の結果をMasterに返します。
4.トレーニングが完了。ここでMasterにはトレーニングを終えたトレーニングの複製があります。

例えば、以下の図には、5つのWorker（W 1、...、W5）によるパラメータ平均化処理、そしてパラメータ平均化の頻度である1が表示されています。
オフラインでのトレーニングのように、トレーニングデータセットは数多くのサブセットに分割されます（非分散型設定ではミニバッチとして知られています）。各分割にトレーニングを行い、各Workerに分割のサブセットが提供されます。実際には、分割数はトレーニングの設定（Worker数、平均化の頻度、Workerのミニバッチのサイズに基づきます。設定のセクションを参照）に基づいて自動的に決定されます。

![Parameter Averaging](./img/parameter_averaging.png)

## <a name="minimal">必要最低限のコンポーネントを使った例</a>

こちらは、Sparkでネットワークをトレーニングするために最低限必要なコンポーネントのセットです。
ローディングに対する様々なアプローチの詳細についても近々ご紹介します。

```java
    JavaSparkContent sc = ...;
    JavaRDD<DataSet> trainingData = ...;
    MultiLayerConfiguration networkConfig = ...;

    //TrainingMasterインスタンスを作成する。
    int examplesPerDataSetObject = 1;
    TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            （その他の設定オプション）
            .build();

    //SparkDl4jMultiLayerインスタンスを作成する。
    SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, networkConfig, trainingMaster);

    //トレーニングデータを使ってネットワークを当てはめる。
    sparkNetwork.fit(trainingData);
```

## <a name="configuring">TrainingMasterの設定</a>

DL4JのTrainingMasterは、SparkDl4jMultiLayerやSparkComputationGraphに使用する複数の異なるトレーニングが実装できる抽象化（インターフェイス）です。 

現在、DL4JにはParameterAveragingTrainingMaster一つのみが実装されています。上図でご紹介したパラメータ平均化処理を実装しています。
これを作成するには、以下のビルダーパターンを使用してください。

```java
    TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(int dataSetObjectSize)
            ...（自分の設定をここに入れます。）
            .build();
```

ParameterAveragingTrainingMasterは、どのようにしてトレーニングを実行するかを管理する数多くの設定オプションを設定します。

* **dataSetObjectSize**:必須オプション。これは、ビルダーコンストラクタで指定されています。この値は各データセットオブジェクトにいくつexampleがあるかを指定します。通常は以下のような規則が適用されます。
    * 事前処理されたデータセットのオブジェクトを使ってトレーニングしているのであれば、事前処理されたデータセットのサイズになります。
    * Stringsから直接トレーニングしている場合（例えば、数多くの段階を踏んでCSVデータから```RDD<DataSet>```まで）は、通常1になります。
* **batchSizePerWorker**:各Workerのミニバッチサイズを管理します。単一の機械でトレーニングをする際に使用されるミニバッチサイズと類似しています。別の言い方をすると、各Workerにある各パラメータのアップデートに使用されるexampleの数に当たります。
* **averagingFrequency**:batchSizePerWorkerでのサイズでのミニバッチの数を基にし、パラメータが平均化され、再分配される頻度を管理します。通常は以下のようなことが言えます。
    * 平均化期間が短いと（例えば、averagingFrequency=1）、非効率的になります（過多のネットワークのコミュニケーション、初期化のオーバーヘッド、計算に関連）。
    * 平均化期間が長いと（例えば、averagingFrequency = 200）、パフォーマンスが低下します（各Workerのインスタンス内のパラメータが大幅に異なる可能性があります）。
    * 5～10の範囲でのミニバッチでの平均化期間が通常、安全なデフォルト設定です。
* **workerPrefetchNumBatches**:SparkのWorkerは、データの読み込みを待たなくてもいいように非同期的に数多くのミニバッチ（データセットのオブジェクト）を先読みすることが可能です。
    * この値を0に設定すると先読みが無効になります。
    * 多くの場合、2が実用的なデフォルトの値です。これよりもっと大きな値は、多くの状況で役立ちません（しかしメモリは多く使用します）。
* **saveUpdater**:DL4Jでは、トレーニング方法のmomentum、RMSProp、AdaGradなどは「アップデーター」として知られています。ほとんどのこれらのアップデーターには内部に履歴や状態があります。
    * saveUpdaterがtrueに設定されている場合、アップデーターの状態（各Workerで）は、平均化され、パラメータと共にMasterに返されます。現在のアップデーターの状態もMasterからWorkerに再分配されます。これによって時間も掛かり、ネットワークトラフィックも増しますが、トレーニングの結果が改善する可能性があります。
    * saveUpdaterがfalseに設定されていると、アップデーターの状態が破棄され、アップデーターは各Workerにおいてリセットされ、再初期化されます。
* **repartition**:データがいつ再分割されるべきかに関する構成設定ParameterAveragingTrainingMasterは、mapParititonsの演算を行います。したがって、パーティション数（及び各パーティションの値）は正しいクラスタ利用に大いに関連しているのです。しかし、再分割は自由な演算ではありません。必ずネットワークを横断して複製しなければならないデータがあるからです。以下のようなオプションがあります。
    * Always:デフォルトのオプション。正しいパーティション数を確保するためにデータを再分割する。
    * Never:どんなにパーティションのバランスが取れていなくてもデータを再分割しない。
    * NumPartitionsWorkersDiffers:パーティション数とWorker数（全コア数）が異なる場合のみ再分割する。しかし、パーティション数が全コア数と同じであっても、正確な数のDataSetオブジェクトが各パーティションに存在するということを保証するものではありません。中には他のパーティションよりはるかに大きい、または小さいパーティションもあります。
* **repartitionStrategy**:どの再分割を行うべきかについてのストラテジー。
    * SparkDefault:Sparkによって使用される標準的な再分割ストラテジー。基本的に最初のRDD(Resilient Distributed Dataset、耐障害性分散データセット）の中にある各オブジェクトはN個のRDDにランダムにマッピングされます。このため、パーティションはバランスが最適な状態にない場合があります。特に問題となるのは、前処理されたデータセットオブジェクトに使用されたり、平均化期間が頻繁な（単にランダムサンプリングのバリエ―ションが原因で）小さめのRDDの場合です。
    * Balanced:これはDL4Jが設定したカスタム再分割ストラテジーです。SparkDefaultオプションと比べて各パーティションのバランスがもっと取れている（オブジェクトの数という点において）ことを確保しようとします。しかし、実際は、場合によってはこれにはさらにカウント動作が必要になります（小規模なネットワークまたはミニバッチ一つにつき少しの計算）。より優れた再分割を行うことによって生じるオーバ―ヘッドにその利点が見合わない可能性があります。   
    



## <a name="dependencies">Sparkでトレーニングをする場合の依存関係</a>

DL4JをSparkで使用するには、deeplearning4j-sparkの依存関係を含める必要があります。

```
        <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>dl4j-spark_${scala.binary.version}</artifactId>
        <version>${dl4j.version}</version>
        </dependency>
```

```_${scala.binary.version}```は```_2.10```または```_2.11```であるべきで、ご利用のSparkのバージョンと一致していなければならないことにご注意ください。 


## <a name="examples">Sparkを使った例のリポジトリ</a>

[Deeplearning4j examplesリポジトリ](https://github.com/deeplearning4j/dl4j-examples)（[以前の例はこちら](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples))には数多くのSparkの例があります。


## <a name="memoryyarn">YARN上でSparkもメモリを設定</a>

Apache Hadoop YARNは、Hadoopクラスター向けによく使用されるリソースマネジャーです（[Apache Mesos](http://mesos.apache.org/)はその代用となるものです）。
Spark submit経由でジョブを提出するとき、設定オプションを少し指定する必要があります。例えばエグゼキュータ数、エグゼキュータ一つにつきのコア数、各エグゼキュータのメモリ量などです。

Sparkのトレーニングで最高のパフォーマンスをDL4Jが達成するには（そしてメモリ制限の超過を防止するには）、さらにメモリ設定する必要があります。このセクションでは、なぜこれが必要なのか、そしてそれを実際にする方法をご説明します。

### <a name="memory1">Deeplearning4j (及びND4J）のメモリ管理方法</a>

Deeplearning4jは数値計算ライブラリのND4Jを基盤として構築されます。DL4Jでのニューラルネットワークの実装は、ND4Jのマトリックスとベクトル演算を使用して構築されます。

ある1つのカギとなるND4Jの設計面は、オフ・ヒープ（Off-heap）メモリ管理を利用している、というところにあります。これはどういうことかと言うと、ND4JによってINDArraysに割り当てられたメモリはJVMのヒープ上に割り当てられない、ということを意味します（この点で標準的なJavaオブジェクトとは異なる）。その代わりに、JVM外部にある別のメモリプールに割り当てられます。このメモリ管理は、[JavaCPP](https://github.com/bytedeco/javacpp)を使用して実装されます。

オフ・ヒープメモリ管理には数々の利点があります。
それらのうちでも最も注目に値するものは、パフフォーマンス性の高いネイティブ（c++）コードを効率的に数値演算で使用できることです（OpenBLASやIntel MKLなどのBLASライブラリ、C++ライブラリの[Libnd4j](https://github.com/deeplearning4j/libnd4j)を使用します）。オフ・ヒープメモリ管理は、CUDAを使ったGPU演算を効率的にするためにも必要です。メモリがJVMヒープ上に割り当てられると（他のJVM BLAS実装のように）、まずはデータをJVMから複製し、演算を行い、結果を複製して返します。各演算には、メモリと時間のオーバーヘッドの両方を追加します。その代わり、ND4Jは単に数値演算のデータへのポインタを渡すことができ、データのコピーの問題を完全に防止できるようにします。

ここで理解しておくべき重要な点はオン・ヒープ（on-heap、JVMヒープ）メモリとオフ・ヒープ（ND4J/JavaCPP）は二つの異なるメモリプールであるということです。個々のサイズを別々にデフォルトで設定することは可能ですが、JavaCPPはオフ・ヒープメモリの割り当てをRuntime.maxMemory()設定と同じぐらい増大させることができるのです。（[コード](https://github.com/bytedeco/javacpp/blob/master/src/main/java/org/bytedeco/javacpp/Pointer.java)をご覧ください。）このデフォルトは基本的にJavaのメモリ設定に使用されるJVMのXmxメモリ設定サイズと同じです。

JavaCPPが割り当てることのできるオフ・ヒープメモリ最大量を手動で管理するには、```org.bytedeco.javacpp.maxbytes```システムプロパティを設定します。ローカルで使用する単一のJVMの場合、オフ・ヒープ割り当てを1GBに制限するために```-Dorg.bytedeco.javacpp.maxbytes=1073741824```をパスします。YARNのSparkでどのように設定するかは後のセクションでご説明しましょう。


### <a name="memory2">YARNのメモリ管理</a>

上述にもあるように、YARNはクラスタリソースを管理します。計算タスクをYARNの管理するクラスタに提出するとき（DLPJのSparkネットワークトレーニングなど）、制限付きリソースプール（メモリ、CPUコア）を自分のジョブ（及びその他のジョブ）に割り当てる仕事の管理責務を担うのはYARNです。YARNとそのリソース割り当ての詳細については、 [こちら](http://blog.cloudera.com/blog/2015/09/untangling-apache-hadoop-yarn-part-1/)、そして[こちら](http://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-2/)をご覧ください。


我々の目的のキーポイントは以下の通りです。

* YARN上のジョブはコンテナ内で実行されます。各ジョブのメモリ量は決まったものに定められています。
* YARN上のコンテナに割り当てられるメモリ量はユーザーが要求したオン・ヒープ（つまりJVMメモリサイズ）とオフ・ヒープ(YARN用語ではメモリのオーバーヘッド）の総和です。
* タスクがコンテナに割り当てられたメモリ量を超えると、YARNはコンテナを消し、コンテナ内でエグゼキュータが実行されます。正確な動きはYARNの設定によって決まります。
* コンテナメモリの制限を超えるプログラムはオフ・ヒープメモリが原因でそのようにします。オン・ヒープ（JVM）メモリの最高量はSpark submitによってローンチパラメータとして固定されています。


YARNがどのぐらいの量のメモリをコンテナに割り当てるかの管理については二つの重要な設定オプションがあります。

1.```spark.executor.memory```:これは標準的なJVMメモリ割り当てです。単一のJVMのXmx設定に類似しています。
2.```spark.yarn.executor.memoryOverhead```:これはコンテナに割り当てられた「余分な」メモリー量です。JVMに割り当てられておらず、オフ・ヒープメモリ（ND4J/JavaCPPを含む）を使用するコードが使用することができます。

デフォルトで、```spark.yarn.executor.memoryOverhead```設定はエグゼキュータメモリの10%、最小量は384MBとなっています。
詳細については、[Apache SparkのYARNに関する情報サイト](http://spark.apache.org/docs/latest/running-on-yarn.html)をお読みください。

ND4Jが広範囲にわたってオフ・ヒープメモリを使用しているので、一般にはSparkでトレーニングをしている時はメモリオーバーヘッド設定を増加させることが必要です。


### <a name="memory3">Deeplearning4jのYARN上でのSparkのトレーニング用にメモリ設定</a>

前のセクションの内容をまとめると、YARN経由のSpark上でニューラルネットワークのトレーニングを実行している間に以下のことを行わなければならない、ということになります。

1.```spark.executor.memory```を使用してエグゼキュータJVMメモリ量を指定する。
2.```spark.yarn.executor.memoryOverhead```を使用してYARNコンテナのメモリオーバーヘッドを指定します。
3.オフ・ヒープメモリの使用が許可された量を```org.bytedeco.javacpp.maxbytes```システムプロパティを使ってND4J/JavaCPPに知らせます。

これらの値を設定するとき、念頭に置いておかねばならないことがあります。
まず最初に、```spark.executor.memory```と```spark.yarn.executor.memoryOverhead```の総和はYARNが単一のコンテナに割り当てる最大メモリ量より少なくなくてはなりません。この制限は通常YARN設定またはYARNのリソースマネジャーのサイトのユーザーインターフェイスで確認できます。この制限を超えると、YARNはこのジョブを拒否します。

次に、```org.bytedeco.javacpp.maxbytes```の値は必ず```spark.yarn.executor.memoryOverhead```より少なくなっていなければなりません。デフォルト設定でmemoryOverheadの設定がエグゼキュータメモリの10%であることを思い出してください。これはJVM自体（そしてその他のライブラリもおそらく同様に）がいくらかのオフヒープメモリを必要としているからです。したがって、JavaCPPがJVMが割り当てたものでないメモリを使い切らないようにしておくことが必要です。  

最後に、DL4J/ND4Jがデータ、パラメータ、活性化用にオフヒープメモリを利用するために、JVM（つまりexecutor.memory）に割り当て量を他の場合と比べて減らすことができます。とはいえ、Spark自体（そしてその他の使用しているライブラリ）に十分なJVMメモリが必要ですので、あまり多く減らす訳にはいきません。

以下は一例です。Sparkのトレーニングを実行しており、メモリを以下のように設定したいとします。

* エグゼキュータ4つ、それぞれコア数は8
* メモリが割り当てることのできる最大コンテナメモリサイズ：11GB
* JVM（エグゼキュータとドライバ）メモリサイズ：4GB
* ND4J/JavaCPPのオフヒープメモリサイズ（エグゼキュータとドライバ）：5GB
* 余分なオフヒープメモリサイズ：1GB

オフヒープメモリの総サイズは5+1=6GBです。総メモリサイズは（(JVM + オフヒープ/オーバーヘッド）4+6=10GBです。これはYARNの最大割り当て量の11GBよりも低い量です。JavaCPPメモリはバイトで指定されており、5GBとは5,368,709,120バイトであり、YARNメモリのオーバーヘッドはMBで指定されており、 6GBとは6,144MBであることにご注意ください。

Spark submitの引数は以下のように指定されます。

```
--class my.class.name.here --num-executors 4 --executor-cores 8 --executor-memory 4G --driver-memory 4G --conf "spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5368709120" --conf "spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5368709120" --conf spark.yarn.executor.memoryOverhead=6144
```


## <a name="kryo">KryoシリアライゼーションのDeeplearning4jでの使用</a>

Kryoは、 Apache Sparkでよく使用されるシリアライザーションライブラリです。オブジェクトをシリアライズするのに掛かる時間を減らすことによってパフォーマンスを向上させることを提案しています。
しかし、ND4Jにあるオフヒープデータ構造にKryoを適用するのは困難です。KryoシリアライゼーションをND4Jで使用するにはSparkの設定をさらに追加する必要があります。
Kyroが正常に設定されていなければ、シリアライゼーションが間違っているために、INDArrayフィールドのいくつかでNullPointerExceptionsが発生する可能性があります。

Kryoを使用するには、適切な [nd4j-kryo 依存関係](http://search.maven.org/#search%7Cga%7C1%7Cnd4j-kryo)を追加し、Nd4j Kryoのレジストレータを使用するために以下のようにSpark設定を設定します。

```
    SparkConf conf = new SparkConf();
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    conf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
```

Deeplearning4jのSparkDl4jMultiLayerまたはSparkComputationGraphクラスを使用するときはKryoの設定が間違っているときは警告が記録されます。

## <a name="mklemr">Intel MKLを使用してAmazon Elastic MapReduce上でDeeplearning4jを使用</a>

Maven CentalにあるDL4JのリリースはOpenBLASによって分散されます。したがって、このセクションは、Maven Central用のDeeplearning4jのバージョンを使用している方向けではありません。

DL4JがBLASライブラリとしてIntel MKL（Intel Math Kernel Library）のソースから構築されている場合、いくつかの設定を追加し、EMR（Amazon Elastic MapReduce）で使用できるようにしなければなりません。
Intel MKLを使用するためにEMRでクラスタを作成しているとき、追加の設定を行う必要があります。

Create Cluster（クラスタの作成）-> Advanced Options（詳細オプション）-> Edit Software Settings（ソフトウェア設定のエディット）へと行き、以下を追加してください。

```
[
    {
        "Classification":"hadoop-env", 
        "Configurations":[
            {
                "Classification":"export", 
                "Configurations":[], 
                "Properties":{
                    "MKL_THREADING_LAYER":"GNU",
                    "LD_PRELOAD":"/usr/lib64/libgomp.so.1"
                }
            }
        ],
        "Properties":{}
    }
]
```

## リソース

* [Deeplearning4j Examples Repo（Deeplearning4jの例のリポジトリ）](https://github.com/deeplearning4j/dl4j-examples)
* ND4S:[N-Dimensional Arrays for Scala（スカラーのN次元配列）](https://github.com/deeplearning4j/nd4s)
* [ND4J, Scala & Scientific Computing（ND4J、スカラー、科学的計算）](http://nd4j.org/scala.html)
* [Intro to Iterative Reduce（繰り返し処理による縮小とは）](./iterativereduce)
