---
title: "Deeplearning4jをGPU上で使う"
layout: ja-default
---

# Deeplearning4jをGPU上で使う

Deeplearning4jは分散GPUでもネイティブでも使用することができます。ローカルの単一GPUであるNVIDIA Tesla、Titan、GeForce GTX、NVIDIA GRID GPUのクラウドなどで使用できます。 

GPU上でニューラルネットワークをトレーニングするには、POM.xmlファイルに1つ変更を行う必要があります。[クイックスタート](./quickstart)でご説明しましたが、POMファイルはCPUで動作するようにデフォルト設定されています。以下のような感じです。

        <name>DeepLearning4j Examples Parent</name>
        <description>Examples of training different data sets</description>
        <properties>
            <nd4j.backend>nd4j-native-platform</nd4j.backend>

今回は、DeeplearningをGPU上で使用したいので、依存関係の`nd4j`の次行である`artifactId`にある`nd4j-native-platform`を`nd4j-cuda-7.5-platform`に変更します。以下のようになります。

            <nd4j.backend>nd4j-cuda-7.5-platform</<nd4j.backend>

ND4Jは、Deeplearning4jが使用する数値計算エンジンです。ND4Jにはいわゆる「バックエンド」というものがあります。あるいは様々な種類のハードウェアを使用します。[Deeplearning4j Gitter channel](https://gitter.im/deeplearning4j/deeplearning4j)では、バックエンド、チップに関連したパッケージについての情報交換を行うところです。バックエンドでは、ハードウェア上の最適化が行われています。

## トラブルシューティング

複数のGPUを持っていても、ご使用のシステムで使用できるGPUが一つに限られている場合は、以下の方法で解決しないか試してみてください。`main()`メソッドの最初の行に`CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);`を追加するだけです。

<p align="center">
<a href="./quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Get Started With Deeplearning4j on GPUs</a>
</p>


## 複数のGPUでのデータ並列処理

お使いのシステムに複数のGPUがインストールされている場合、データ並行処理モードでモデルをトレーニングすることが可能です。この目的に使用するシンプルなラッパーがあります。

以下のようなものを考えてみてください。

        ParallelWrapper wrapper = new ParallelWrapper.Builder(YourExistingModel)
            .prefetchBuffer(24)
            .workers(4)
            .averagingFrequency(1)
            .reportScoreAfterAveraging(true)
            .useLegacyAveraging(false)
            .build();

ParallelWrapperは既存のモデルを主な引数として受け取り、並行してトレーニングします。GPUの場合は、Worker数をGPU数と同数またはそれ以上にしておくのがいいでしょう。正確な値はタスクや使用可能なハードウェアによって異なるため、調整が必要となります。

`ParallelWrapper`内で、最初のモデルが複製され、Workerは、各々のモデルをトレーニングします。`averagingFrequency(X)`によって指定される*X*回のイテレーションの後は常に、すべてのモデルが平均化され、その後にトレーニングが行われます。 

データの並列処理には、学習率が高い方がいいでしょう。+20%辺りが開始時には望ましい値でしょう。

## HALFデータ型

半精度数学を使用できるなら（通常ニューラルネットはこれが可能です）、これをデータタイプとして有効化すると以下のような利点があります。

* GPUの使用するRAMが半分になる。
* メモリ集約的な作業において最大200%までのパフォーマンスを得ることができます。ただし、実際のパフォーマンスの向上はタスクや使用されるハードウェアによって異なります。

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

お使いのアプリケーションの最初の行にこの呼び出しを配置すると、後に続くすべての割当/計算はHALFデータタイプを使って行われます。 

## 大きめのグリッド

ほとんどのGPUの場合、デフォルト値で問題ないのですが、お使いのハードウェアがハイエンドで、既にデータが大量の場合、大きめのグリッド/ブロック制限を試すといいかもしれません。以下のようにします。

    CudaEnvironment.getInstance().getConfiguration()
          .setMaximumGridSize(512)
          .setMaximumBlockSize(512);

これにより、小規模の演算でも特定のグリッド次元しか使用できなくなるわけではありませんが、理論上の制限が設けられます。 

## キャッシュサイズを増大させる

ND4JはJavaを使用しているため、CUDAのバックエンドのキャッシュサイズは非常に重要です。パフォーマンスを大幅に増減させることができるのです。RAMがたくさんあるのであれば、大きめのキャッシュを設定するといいでしょう。

以下のようにするといいでしょう。

        CudaEnvironment.getInstance().getConfiguration()
        .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
        .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
        .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
        .setMaximumHostCache(6L * 1024 * 1024 * 1024L);

このコードにより、GPU RAMの最大6GBまでのキャッシュが可能になります（これは十分な割り当てを行うというわけでありません）。そして、個々のキャッシュしたメモリチャンクはホストもGPUも最大1GBまでのメモリサイズが可能です。 

Nd4jのキャッシュには「再利用」というパラダイムがあるため、値が高いと良くないわけではありません。お使いのアプリケーションに割り当てられたメモリチャンクのみが将来再利用されます。
