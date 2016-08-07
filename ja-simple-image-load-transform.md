---
title: ディープニューラルネットワークに画像を読み込むためにカスタマイズされたデータパイプライン
layout: ja-default
---

# 画像向け等にカスタマイズされたデータパイプライン

Deeplearning4jのexamplesに使用する標準データセットは抽象化されているため、データパイプラインに全く障害が生じません。しかし、実際のユーザーが最初に手を付けるのは生の乱雑なデータであるため、前処理やベクトル化を行い、ニューラルネットワークがクラスタリングや分類を行うための訓練をする必要があります。 

*DataVec*は、弊社の機械学習ベクトル化ライブラリで、ニューラルネットワークが学習できるデータを準備するための方法をカスタマイズするのに役に立ちます。([DataVecのJavadocはこちらをご参照ください。](http://deeplearning4j.org/datavecdoc/).)

こちらのチュートリアルでは、画像のデータセットの読み込み方法、変換の実行についてご説明します。ここでは簡単に*Oxford flower dataset（オックスフォードの花のデータセット）*の3クラスの画像10個のみを使用します。下記のコードは参照用のみとしてご利用いただき、コピー・ペーストはご遠慮願います。 
[こちらから全exampleのコードをご利用ください。](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/ImagePipelineExample.java)

## 該当するディレクトリ構造に画像を収納
手短に言うと、データセット内の画像は、ディレクトリのクラス/ラベルにより整理され、これらのクラス/ラベルのディレクトリは、親ディレクトリ内に収められている必要があります。

* データセットをダウンロードします 

* 親ディレクトリを作成します。

* 親ディレクトリ内にラベル/クラス名に対応するサブディレクトリを作成します。

* 該当するクラス/ラベルに属するすべての画像を各々のサブディレクトリに移動します。

一般的なディレクトリの構造は以下のようなものになります。

>                                   parentDir
>                                 /   / | \  \
>                                /   /  |  \  \
>                               /   /   |   \  \
>                              /   /    |    \  \
>                             /   /     |     \  \
>                            /   /      |      \  \
>                      label_0 label_1....label_n-1 label_n


この例では、parentDir（親ディレクトリ）は `$PWD/src/main/resources/DataExamples/ImagePipeline/`と対応しており、サブディレクトリであるlabelA、labelB、labelCにはそれぞれ画像が10個づつ含まれています。 

## 画像を読み込む前に詳細事項を指定
* それぞれ別のディレクトリにあるラベルが付与された画像の親ディレクトリのパスを指定します。
 
~~~java
File parentDir = new File(System.getProperty("user.dir"), "src/main/resources/DataExamples/ImagePipeline/");
~~~

* データセットをテストや訓練に分割する際に使用するために、適用可能な拡張子と乱数発生器を指定します。 

~~~java
FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
~~~

* 手動でラベルを指定せずに済ますためにラベルメーカーを指定します。これによりサブディレクトリの名前がそのままラベル/クラスの名前に適用されます。

~~~java
ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
~~~

* パスフィルターを指定し、各クラスに読み込む最小/最大ケースを微調整コントロールします。以下はその基本バージョンです。詳細はjavadocsをご参照ください

~~~java
BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
~~~

* テストや訓練への分割をします。この例では80%-20%と指定しています。

~~~java
InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
InputSplit trainData = filesInDirSplit[0];
InputSplit testData = filesInDirSplit[1];
~~~

## 画像パイプライン変換の詳細事項を指定

* 画像記録リーダーで全体のサイズ変更をしたいデータセットの高さと幅を指定します。 

~~~java
ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
~~~
*データセットの画像は同じサイズである必要はありません。* DataVecがこの作業を代行してくれます。この例にもあるように、画像のサイズはすべて異なり、以下に指定された高さと幅に変更されています。

* サイズ変更の指定を行います。

ニューラルネットワークの利点は、手動で機能のエンジニアリングをする必要がないというところです。しかし、人為的にサイズを大きくするために画像変換させると役に立つことがあります。例えば、Kaggleのコンテスト参加で勝利を狙う場合などです <http://benanne.github.io/2014/04/05/galaxy-zoo.html>。また、画像内の必要な部分以外のみを残してその他の部分をトリミングしたいこともあります。例えば、顔面部分を検知し、その他の部分をトリミングしてサイズ調整するなどです。DataVecには、OpenCVから導入された機能/強力な特徴がすべて備えられています。以下は、画像を反転させ、表示するために使用する基本的なexampleです。

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen,new FlipImageTransform(), new ShowImageTransform("After transform"));
~~~

変換命令を以下のように連鎖させることができます。使用したい機能を自動設定するクラスを入力します。

~~~java
ImageTransform transform = new MultiImageTransform(randNumGen, new CropImageTransform(10), new FlipImageTransform(),new ScaleImageTransform(10), new WarpImageTransform(10));
~~~

* 訓練するデータと変換連鎖で記録リーダーを開始します。

~~~java
recordReader.initialize(trainData,transform);
~~~

## 適合調整のためのデータセット
Deeplearning4jのニューラルネットワークは、適合を調整させるためにもデータセットまたはデータセットのイテレータを使用します。これらは弊社のフレームワークの基盤としている概念です。イテレータの使用方法については、他のexampleをご参照ください。以下の方法により画像記録リーダーからデータセットのデータセットを構築します。

~~~java
DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
~~~

DataSetIteratorは入力データベースをrecordReader経由でイテレートします。各イテレーションにつき、新しいexampleを1つ、またはそれ以上取り入れ、それらをニューラルネットワークが使用できるDataSetオブジェクトに読み込みます。


