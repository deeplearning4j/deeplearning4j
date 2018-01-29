---
title: Stacked Denoising AutoEncoders
layout: default
---

# 積層ノイズ除去オートエンコーダー

積層ノイズ除去オートエンコーダーとノイズ除去オートエンコーダーとの関係は、[ディープ・ビリーフ・ネットワーク](https://deeplearning4j.org/deepbeliefnetwork.html)と[制限付きボルツマン・マシン](https://deeplearning4j.org/ja/restrictedboltzmannmachine)の関係に似ています。積層ノイズ除去オートエンコーダーを含めて、一般にディープラーニングの主な機能は、入力がある度に各層で教師なしの事前トレーニングが行われるということです。前層からの入力の特徴を選択し、抽出するためにいったん各層が事前トレーニングを終えると、次に教師付きの微調整を行うことができます。 

ノイズ除去オートエンコーダーにおける確率的な破壊について少しお話ししておきましょう。ノイズ除去オートエンコーダーは、データをシャッフルし、復元を試みることによりデータについて学習します。このシャッフリングを行う行為がノイズに当たり、ネットワークの仕事は入力を分類できるノイズ内の特徴を認識することです。ネットワークをトレーニングすると、ネットワークはモデルを生成し、損失関数を使ってモデルとベンチマークとの距離を測定します。損失関数を最小化しようとする試みには、正解とされたものに最も近くにモデルを近付けることができる入力が見つかるまで、シャッフルされた入力をリサンプルし、データを復元する作業が含まれます。 

連続的なリサンプリングは、処理するデータをランダムに提供する生成的モデルを基としています。これはマルコフ連鎖、またより具体的にはマルコフ連鎖モンテカルロアルゴリズムとして知られていますが、このアルゴリズムはデータセットをステップスルーし、さらに複雑な特徴を構築するのに使用できるインジケーターの代表的サンプリングを探します。

Deeplearning4jでは、積層ノイズ除去オートエンコーダは隠れ層にオートエンコーダのある`MultiLayerNetwork`を作成して構築されます。これらのオートエンコーダには`corruptionLevel`があります。これが「ノイズ」であり、ニューラルネットワークは信号のノイズを除去をすることを学習します。`pretrain`がtrueに設定されていることに注意してください。

同じトークンによって、各隠れ層に制限付きボルツマン・マシンを持つ`MultiLayerNetwork`としてディープ・ビリーフ・ネットワークが作成されます。より一般的に言うと、Deeplearning4jは様々なディープ・ニューラル・ネットワークを構築する制限付きボルツマンマシンやオートエンコーダなどの「プリミティブ」を使用していると考えていいでしょう。

## 使用するコード


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(4)
           .layer(0, new AutoEncoder.Builder().nIn(numRows * numColumns).nOut(500)
                   .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                   .corruptionLevel(0.3)
                   .build())
                .layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)

                        .build())
                .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
                .build();

### <a name="beginner">その他のDeeplearning4j のチュートリアル</a>
* [ディープニューラルネットワークについて](https://deeplearning4j.org/ja/neuralnet-overview)
* [再帰型ネットワークと長・短期記憶についての初心者ガイド](https://deeplearning4j.org/ja/lstm)
* [Word2Vecとは？](https://deeplearning4j.org/ja/word2vec)
* [制限付きボルツマンマシンの初心者向けガイド](https://deeplearning4j.org/ja/restrictedboltzmannmachine)
* [固有ベクトル、主成分分析、共分散、エントロピー入門](https://deeplearning4j.org/ja/eigenvector)
* [ニューラルネットワークを回帰に使用](https://deeplearning4j.org/ja/linear-regression)
* [畳み込みネットワーク](https://deeplearning4j.org/ja/convolutionalnets)
