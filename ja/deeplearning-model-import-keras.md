----
title:Deeplearning4j Keras Model Import
layout: default
---

# KerasからDeeplearning4jにモデルをインポート

*モデルのインポートは新機能ですのでご注意ください。2017年2月以降は、ユーザーの皆様は最新バージョンをご使用になるか、問題作成を行ったり不具合を報告する前にマスターからローカルに構築することをお勧めします。 

`deeplearning4j-modelimport`モジュールは、最初に設定され、トレーニングされたニューラルネットワークモデルをインポートするルーチンを提供します。
また、これには[Keras](https://keras.io/)と言って、Deeplearning4jの最上部に抽象化層を提供する人気のあるPythonのディープラーニングライブラリ、
[Theano](http://deeplearning.net/software/theano/)、[TensorFlow](https://www.tensorflow.org)
のバックエンドを使います。Kerasモデルの保存について詳細を知りたい方は、Kerasの[FAQページ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)をご覧ください。Deeplearning4jのKerasを使ったPythonのAPI)の詳細は[こちらのリンク](https://github.com/crockpotveggies/dl4j-examples/tree/keras-examples/dl4j-keras-examples)をご覧ください。

![Model Import Schema](./img/model-import-keras.png)

`IncompatibleKerasConfigurationException`というメッセージは、Deeplearning4jが現在対応していない
Kerasモデル設定をインポートしようとしていることを示します（モデルのインポートがそれに対応していない、またはDL4Jがそのモデル、層、特徴を実装しないため）。

いったんモデルをインポートしたら、さらなる保存や再読み込みには弊社のモデル・シリアライザクラスをお勧めします。 

この詳細は、[DL4JのGitterチャンネル](https://gitter.im/deeplearning4j/deeplearning4j)をお読みください。
あるいは、[Github経由で特徴のリクエスト](https://github.com/deeplearning4j/deeplearning4j/issues)を申請する、
または必要な変更を書いて弊社にプルリクエストで送信していただければ、この機能の追加のお手伝いをいたします。


頻繁にある更新については、モデルのインポートモジュール*と*このページの両方でチェックしましょう！

## 人気の高いモデルのサポート

VGG16やその他の事前にトレーニングされたモデルはデモンストレーションを目的として、また特定のユースケース用に再トレーニングするために広く使用されています。弊社は、VGG16のインポートのサポートと共にデータの取り込みが行えるようにデータを正常にフォーマットし、正常化するヘルパー関数、そして数値出力をラベル化したテキストクラスに変換するヘルパー関数の提供を行っています。  

## DeepLearning4JモデルのZoo

DeepLearning4jは、事前にトレーニングされたKerasのモデルをインポートするだけでなく、弊社のモデルzooへのモデル追加を行っています。 

## IDEを設定し、modelimportクラスにアクセス

次の依存関係を追加してPom.xmlを編集してください。

```
<dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-modelimport</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
```

## 使用できる方法

Kerasモデルのインポート機能を使用すると以下のオプションがあります。Kerasには、SequentialとFunctionalの2種類のネットワークがあることに注意してください。KerasのSequentialモデルはDeepLeanring4JのMultiLayerNetworkに相当します。KerasのfunctionalモデルはDeepLearning4JのComputation Graphに相当します。  

## モデルの設定

すべてのモデルがサポートされているわけではないということに注意してください。しかし、最も有益で広く使用されているネットワークをインポートするのが弊社の目標です。

この機能を使用するには、KerasにあるモデルをJSONファイルに保存します。使用可能なDeepLEarning4Jのオプションは以下の通りです。 

* Sequentialモデル 
* Sequentialモデル及びさらなるトレーニングが可能なアップデーター
* Functionalモデル
* Functionalモデル及びさらなるトレーニングが可能なアップデーター

### コードの紹介

* model.to_json()と共にKerasに保存されたSequentialモデル設定のインポート

```
MultiLayerNetworkConfiguration modelConfig = KerasModelImport.importKerasSequentialConfiguration（"JSONファイルへのパス）

```

* model.to_json()と共にKerasに保存されたComputationGraph設定のインポート

```
ComputationGraphConfiguration computationGraphConfig = KerasModelImport.importKerasModelConfiguration（"JSONファイルへのパス）

```






## モデルの設定、及びKerasでトレーニングされたモデルからの保存された重み

この場合、JSONの設定とKerasでトレーニングされたモデルからの重みの両方を保存します。この重みはH5フォーマットのファイルに保存されます。Kerasでは、重みとモデル設定を単一のH5ファイル、または別ファイルに保存することができます。 

### コードの紹介

* Sequentialモデルの単一ファイル

```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights（"JSONファイルへのパス）

```

ネットワークに推論作業を開始させるには、元のデータと同じようにフォーマット、変換、正規化が行われた入力データに通過させ、network.outputを呼び出します。

* Sequentialモデル用の1ファイル、重みの用の1ファイル 


```
MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("JSONファイルへのパス"、"H5ファイルへのパス")

```

## その他のオプション

modelimport機能にはenforceTrainingConfigパラメータが一つ含まれています。 

推論作業用のみに事前トレーニングされたモデルをインポートしたい場合、enforceTrainingConfig=falseに設定するのがいいでしょう。サポートされていないトレーニング用のみの設定をすると、警告が生成されますが、モデルのインポートは行われます。

トレーニング用にモデルをインポートし、生成されたモデルが可能な限りKerasモデルに一致していることを確認したい場合は、enforceTrainingConfig = trueに設定するのがいいでしょう。この場合、サポートされていないトレーニング用のみの設定はUnsupportedKerasConfigurationExceptionを投げ、モデルのインポートを停止します。



## Kerasモデルのインポート

こちらは[ビデオのチュートリアル](https://www.youtube.com/embed/bI1aR1Tj2DM)です。このビデオではKerasモデルをDeeplearning4jに読み込むのに使用可能なコードを紹介し、使用可能なネットワークを立証しています。このビデオでは、講師のトム・ハンロン氏がエクスポートされ、Deeplearning4jに読み込まれたシンプルなIrisデータの分類器についての概要を説明しています。この分類器はTheanoがバックエンドであるKerasに構築されたものです。

<iframe width="560" height="315" src="https://www.youtube.com/embed/bI1aR1Tj2DM" frameborder="0" allowfullscreen></iframe>

このビデオが見れない場合は、こちらをクリックすると、[YouTubeで視聴](https://www.youtube.com/embed/bI1aR1Tj2DM)することができます。

## なぜKerasなのか？

Kerasは抽象化層で、TheanoやTensorflowなどのPythonライブラリの最上部にあり、ディープラーニング用に簡単に使えるインターフェイスを提供します。 

Theanoのようなフレームワークを設定するには、重み、バイアス、活性化関数、そして入力データがどのような変換で出力するのかを正確に設定しなければなりません。 
その上、バックプロパゲーションにも対応し、その重みとバイアスを更新しなければなりません。Kerasはそれらすべてをwrapします。これらの計算と更新を包含する既成の層が提供されます。

Kerasで設定する唯一のものは、入力のシェイプ、出力のシェイプ、そして損失をどのように計算したいか、ということだけです。Kerasによって、すべての層が正しいサイズで、エラーにバックプロパゲーションが適切に行われることが確保されます。バッチングさえも行われます。

詳細情報は[こちら](http://deeplearning4j.org/keras)をお読みください。




