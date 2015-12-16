---
title:
layout: ja-default
---

# Deeplearning4jとは何か?

Deeplearning4j(以下DL4J)はJava, Scalaで書かれた世界初のオープンソース分散deep-learningライブラリになります。HadoopやSparkと
連携することにより研究、調査目的に加えて実際のビジネスに活かせるように作られています。

DL4Jは最先端の技術を非研究者でも手軽にプロトタイピングできるように作られています。DL4JはスケーラブルなシステムでApache 2.0で配布されていますので自由にお使いいただけます。
[Quick Start](/ja-quickstart.html)ガイドをご参照いただければ数分でNeural Networkの学習を始めることができます。

### [Deep learningのユースケース](http://deeplearning4j.org/use_cases.html)

* [顔/画像認識](http://deeplearning4j.org/facial-reconstruction-tutorial.html)
* 音声検索
* 音声の文字化
* スパムフィルタ
* ECサイトにおける不正検出

### DL4Jの主な機能

* 汎用的な[ベクトルクラス](http://nd4j.org/)の実装
* [GPU](http://nd4j.org/gpu_native_backends.html)との連携
* [Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn), [Spark](http://deeplearning4j.org/gpu_aws.html)やAkkaとの親和性とスケーラビリティ

DL4Jは本番環境としては分散モード、マルチスレッドのモードをサポートしつつ、開発用、お試し用としては簡単に利用できるシンプルなシングルスレッドのモードもご利用できます。
クラスタで学習を行えば大量なデータに対して高速に処理を進めることができます。この場合Neural Networkは[逐次的なMapReduce](http://deeplearning4j.org/iterativereduce.html)で学習が進みます。
これはJVM上で動く言語(Java, Scala, Clojureなど)で利用できます。

DL4Jのコンポーネントとして目指しているものは[micro-service architecture](http://microservices.io/patterns/microservices.html)に対応した
初めてのdeep-learningフレームワークになることです。

### DL4Jで実装されているNeural Network

* [Restricted Boltzmann machines](http://deeplearning4j.org/restrictedboltzmannmachine.html)
* [Convolutional Neural Network](http://deeplearning4j.org/convolutionalnets.html)
* Stacked Denoising Autoencoders
* [Recurrent Neural Network/LSTM](http://deeplearning4j.org/recurrentnetwork.html)
* [Deep Belief Network](http://deeplearning4j.org/deepbeliefnetwork.html)
* [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html)
* Recursive Neural Tensor Networks

これらをどのように使うかは[How to Choose a Neural Net](http://deeplearning4j.org/neuralnetworktable.html)もご参照ください。

Deep Neural Networkは[様々なコンテストで驚異的な精度](http://deeplearning4j.org/accuracy.html)を叩きだしています。Neural Networkの導入としては[Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html)をご参照ください。

DL4Jは浅いNeural Networkから様々なタイプのNetworkを作成して学習させるためのツールなのです。
DL4Jを使えばRestricted Boltzmann machineもConvolutional Neural NetworkもSparlやHadoop, CPU, GPUなど環境を問わず利用することができます。

下記にこのエコシステムを作り上げているライブラリ群を図示します。

![ecosystem](http://deeplearning4j.org/img/schematic_overview.png)

deep-learningで学習させるときには非常に多くのパラメタを調整することになります。このドキュメントに記載されていることがきっとその助けとなり
DL4Jが開発者のdeep-learning用のDIYツールとなると思います。

もし何かご質問、ご要望などありましたら遠慮無く[Gitter](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)に参加してください。プレミアムサポートをご要望の方は[Skymind](http://www.skymind.io/contact/)までご連絡ください。

[固有ベクトル、主成分分析、共分散、エントロピー入門](http://deeplearning4j.org/ja-eigenvector.html)
