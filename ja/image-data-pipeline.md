---
title:Customized Data Pipelines for Loading Images Into Deep Neural Networks
layout: default
---

# 画像用等にデータパイプラインをカスタマイズ

Deeplearning4jのサンプルに使用するベンチマークデータセットは抽象化されているため、データパイプラインに全く障害は発生しません。しかし、実際のユーザーが最初に手を付けるのは秩序のない生データであるため、事前処理やベクトル化を行い、ニューラルネットワークがクラスタリングや分類できるよう訓練をする必要があります。 

*DataVec*は、弊社の機械学習ベクトル化ライブラリで、ニューラルネットワークが学習できるデータを準備するための方法をカスタマイズするのに役に立ちます。（Datavec Javadocへは[こちら](http://deeplearning4j.org/datavecdoc/)からアクセスできます。）

このチュートリアルは画像処理に関連したトピックを扱っています。ラベル生成、ベクトル化、画像を取り込むためのニューラルネットワークの設定について説明します。 


## 説明ビデオ

<iframe width="420" height="315" src="https://www.youtube.com/embed/EHHtyRKQIJ0" frameborder="0" allowfullscreen></iframe>

## 全パイプラインのビデオシリーズ

ビデオはシリーズになっており、画像のディレクトリを処理するコード画面を録画したものを使って説明されます。ラベルをパスに基づいて生成し、画像でトレーニングするためのニューラルネットワークを構築します。このシリーズのその他のビデオには、トレーニングしたネットワークの保存や読み込み、インターネットで収集された未知の画像を使ったテストなどについての内容が含まれています。 

シリーズの初回はこちらからご覧ください。

<iframe width="420" height="315" src="https://www.youtube.com/embed/GLC8CIoHDnI" frameborder="0" allowfullscreen></iframe>

## ラベルの読み込み

弊社のサンプルリポジトリにはをParentPathLabelGenerator使ったものがあります。クラスはImagePipelineExample.javaです。 

        File parentDir = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/");
        //親ディレクトリの下位ディレクトリにある「対応する拡張子（allowed extensions）」を持つファイルはファイルをトレーニングとテスト用に分割するとき、再現を可能にするために乱数ジェネレーターが必要になります。
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //ラベルを手動で指定する必要はありません。このクラス（以下にインスタンス生成されたように）は
        //親ディレクトリを解析し、サブディレクトリ名をラベル/クラス名に使用します。
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

## <a name="record">データを反復し、レコードを読み込む</a>

以下のコードを使って、生画像をDL4JとND4Jに対応したフォーマットに変換します。

        // RecordReaderのインスタンスを生成します。画像の縦横の長さを指定します。
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        
        // channelsは画像の色深度を指します。1はグレースケールで3はRGBです。

        // データパスを指し示します。 
        recordReader.initialize(new FileSplit(new File(parentDir)));

RecordReaderはDatavecの中のクラスでバイト指向の入力をレコード指向（数字で固定された固有のIDでインデックスが付与された要素のコレクション）に変換するのを助けます。データのレコードへの変換はベクトル化のプロセスです。レコード自体はベクトルで、その各要素は特徴1つになります。

詳細は[JavaDoc](http://deeplearning4j.org/datavecdoc/org/datavec/image/recordreader/ImageRecordReader.html)をお読みください。 

[ImageRecordReader](https://github.com/deeplearning4j/DataVec/blob/master/datavec-data/datavec-data-image/src/main/java/org/datavec/image/recordreader/ImageRecordReader.java)はRecordReaderのサブクラスで自動で28 x 28画素の画像を取り込むために構築されます。生成される画像の縦横の長さと等しくなるハイパーパラメターの`nIn`さえ必ず調節すれば、ImageRecordReaderに使ったパラメータを変更して自分用にカスタマイズされた画像の寸法に変更することができます。28*28の画像を取り入れるMultiLayerNetwork設定は`.nIn(28 * 28)`となります。

LabelGeneratorが使用されれば、ImageRecordReaderへの呼び出しにはパラメータのlabelGeneratorが含まれます。
`ImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator)`

<!-- ![Alt text](./img/recordreader_extensions.png) 
dl4jのスクリーンショットからこの画像を復元 
-->

DataSetIteratorはリスト内の要素を巡回するDeeplearning4Jのクラスです。イテレーターはデータリストを通過し、各アイテムに順次にアクセスし、現在の要素を指し示すことにより、進捗度をトラッキングします。そして次の要素を指し示すように自己修正します。

        // DataVecからDL4J
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        // パラメータは、DataVecのrecordReader、バッチサイズ、ラベルのインデックスのオフセット、ラベルクラス
        //の総数

DataSetIteratorは各イテレーションにつき、新しいサンプルを1つ（バッチサイズ）以上取り入れ、入力データベースをイテレートします。そしてそれらのサンプルをニューラルネットワークが使用できるDataSet(INDArray)オブジェクトに読み込みます。また、上記のラインは[RecordReaderDataSetIterator](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.java)に画像を28 x 28のグリッド（行列など）でなく要素の直線（ベクトルなど）に変換するよう指示しています。ラベルの設定も明記しています。

`RecordReaderDataSetIterator`はパラメータに、自分の望む特定のrecordReader（画像やサウンド用）やバッチサイズを使用することができます。 教師付き学習には、ラベルインデックスと入力に適用可能なラベル数も使用できます（LFWの場合、ラベル数は5,749）。 

## モデルの設定

以下はニューラルネットワークの設定の一例です。ハイパーパラメータの多くは、[NeuralNetConfiguration Class glossary](./neuralnet-configuration.html)に説明がありますので、ここでは一部の重要な特徴についてのみまとめます。

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/unsupervised/deepbelief/DeepAutoEncoderExample.java?slice=29:71"></script>

* *optimizationAlgo*はLBFGSよりLINE_GRADIENT_DESCENTを頼りにします。 
* 画像の各画素を入力ノードにするために*nIn*は784に設定します。画像の寸法が変更すれば（果画素の総計がある程度変更）、nlnも変更しなければなりません。
* *list*オペレータは4に設定。これは3つのRBM隠れ層と1つの出力層です。1つ以上のRBMがDBNになります。
* **損失関数** は平均二乗誤差（RMSE）に設定。この損失関数は入力を適切に復元するためにネットワークのトレーニングに使用されます。 

## モデルの構築とトレーニング

設定の最後に、buildを呼び出し、ネットワークの設定をMultiLayerNetworkにパスします。

                }).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

性能を表示するイテレーションリスナーを設定してトレーニング中に調整するには、以下のサンプルのどれかを使用します。

        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(10), new GradientPlotterIterationListener(10)));

        または

        network.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

## モデルのトレーニング

データが読み込まれると、モデルの枠組みが構築されるので、モデルがデータに適合するようにトレーニングします。次にバッチサイズに基づくデータによって進めるためにデータのイテレータを呼び出します。毎回、バッチサイズに基づいて特定数のデータを返します。以下のコードはデータセットのイテレーターをループし、そのデータでトレーニングするためにモデルにfitを実行する方法を示したものです。

        // トレーニング
        while(iter.hasNext()){
            DataSet next = iter.next();
            network.fit(next);
        }

## モデルの評価

モデルのトレーニングをした後、その性能のテストや評価をするためにデータを実行します。一般にデータセットを分割して交差検証によりモデルがこれまでに見たことのないデータを使用するのがいいでしょう。ここでは、どのように現在のイテレータを再設定し、評価対象を初期化し、性能の情報を得るためにどのようにしてそこにデータを実行するかをお見せします。

        // テスト用に同じトレーニングデータを使用します。 
        
        iter.reset();
        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }
        
        System.out.println(eval.stats());

交差検証を適用するその他の代替方法は、すべてのデータを読み込み、トレーニングセットとテストセットに分割することです。アイリスのデータセットの規模は大きすぎないため、すべてのデータを読み込み、分割することができます。しかし、多くのデータセットは規模が大きすぎます。このサンプルでの代替アプローチには以下のコードを使用します。

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

テストを分割して大きめのデータセットでトレーニングするには、テストとトレーニングの両方のデータセットをイテレートする必要があります。今のところはユーザーがそれをすることになっています。 

