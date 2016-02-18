---
title: Downloads
layout: ja-default
---

# ND4J

DeepLearning4jは[ND4J](http://nd4j.org/ja-getstarted.html)を数学的なオペレーションツールの核として活用しております。Deeplearning4jを始めるためには、[ND4Jのバックエンド](http://nd4j.org/downloads.html)が必要になります。このバックエンドは、GPUか初期設定されている状況次第で決定されます。

## なぜ交換可能なバックエンドを使うのか?

多くのdeep-learning技術者の方は、並行して作業を進める為そしてにCuda環境を標準設定しているかと思います。しかし、多くの専門家が昔ながらのハードウェアを使うことで、より制限された環境下で作業していることも事実です。このような制限された環境下でも、CPUが解決する問題にdeep-learningを応用することで、スピードを大きく高めることができます。

現在JVM Blasをベースとしたライブラリは、どれも異なったスピーディーな数学的オペレーションを行うための交換可能な環境を持つことに対応しておりません。こういった環境下に問題意識を持ち、私たちはND4Jを創り出しました。交換可能なバックエンドはその唯一の答えです。( [SLF4J](http://slf4j.org/))

加えて、私たちは一般的なAPIを活用して作り上げるmachine-learningのアルゴリズムではこの問題を解決しきれていないと考えております。なぜならば、私たちのアルゴリズムの方が、スピードがはるかに速いからです。

## ダウンロード

ページ下部にあるリンクを通じて、deeplearning4jに必要なものをダウンロードすることができます。

[ND4J バックエンドダウンロード](http://nd4j.org/downloads.html)のように、Deeplearning4jはND4J使うことで、実行することができます。ページ下部にあるリンク先で、それぞれの環境下にあったものをダウンロードください。

# CPUs

## Jblas

### 最新版
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jblas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

## Netlib Blas

### 最新版
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/netlib-blas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jblas/deeplearning4j-dist-bin.zip)

# GPUs

## Jcublas

### 最新版
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/latest/jcublas/deeplearning4j-dist-bin.zip)

### 0.0.3.2.5
* [tar archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.gz)
* [bz2 archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.tar.bz2)
* [zip archive](https://s3.amazonaws.com/dl4j-distribution/releases/0.0.3.2.5/jcublas/deeplearning4j-dist-bin.zip)
