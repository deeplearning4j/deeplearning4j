---
title: 
layout: default
---

# Deeplearning4jとは何か?

Deeplearning4jはJavaを使った世界初のオープンソースdeep-learning libraryになります。この技術は広範囲な情報調査というよりも、ビジネスフィールドでご活用いただけます。具体例としては、 [顔認識](../facial-reconstruction-tutorial.html), 音声認識, スパムメールの識別といった、複雑な情報に対して効果的に活用できます。

Deeplearning4jはGPUを利用し、実行されます。 **[詳しくはコチラ](http://nd4j.org/gpu_native_backends.html)** そして、多様性のある**[n-dimensional array](http://nd4j.org/)**を含んでおります。DL4Jは、最先端のプラグアンドプレイサービスとなることを目指しております。当社が説明している手順で設定していくことで、Hadoopとその他のbig-dataに適した[無限の可能性をもつ](../scaleout.html) deep-learning に関するアーキテクチャを手に入れることができます。このJavaを使ったdeep-learning libraryはニュートラルネットを操作するため必要な、特定の言語を含んでおります。

Deeplearning4jは**distributed deep-learning framework** と通常のdeep-learning frameworkを活用しています。 (一つのスレッドで実行されることもあります。). DL4Jのトレーニングは広大な情報を含むクラスターの中でiterative　reduceというアルゴリズムを通じて行われます。JVMに対応する形で、Java,Scala,Clojureはすべて同様に利用することができます。

このオープンソースのdistributed deep-learning　frameworkdeはデータの入力とニュートラルネットのトレーニング、そして精度の高いアウトプットを生み出すことができます。

それぞれのリンク先のページで、セットアップ方法、サンプルデータといくつかのディープラーニングネットワークを確認することができます。これらはシングルスレッド並びにマルチスレッドを含みます。 [Restricted　Boltzmann　machines](../restrictedboltzmannmachine.html), [deep-belief networks](../deepbeliefnetwork.html), [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html), [Recursive Neural Tensor Networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html), [Convolutional Nets](http://deeplearning4j.org/convolutionalnets.html) と[Stacked　Denoising　Autoencoders](../stackeddenoisingautoencoder.html). 

ニュートラルネットに関する簡単な説明は[コチラ](../overview.html)をご覧ください。簡潔にまとめると、Deeplearning4jは浅いニュートラルネットを組み合わせることで深いニュートラルネットの層を作り出しております。このフレキシブルな仕組みが制限されたBoltzmann machinesやautoencoders、convolutional netsとrecurrent netsを自由に組み合わせることを可能にします。これらの操作は、端末ごとに独立して操作することができます。

deep-learningのネットワークをトレーニングすることに関しては、様々な選択肢があります。 JavaとScala、そしてClojureのプログラマーの方が、自由にそれぞれに合う形に作り上げることができるプログラムがDeeplearning4jです。 ご質問は[当社グーグルグループ](https://groups.google.com/forum/#!forum/deeplearning4j); プレミアムサポートが必要な場合は, [当社ホームページ](http://www.skymind.io/contact.html). [ND4Jについて](http://nd4j.org/) 

![Alt text](../img/logos_8.png)
