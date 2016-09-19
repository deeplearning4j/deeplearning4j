---
title: "Deeplearning 4 j のクイックスタートガイド"
layout: ja-default
redirect_from: /ja-quickstart
---

<!-- Begin Inspectlet Embed Code -->
<script type="text/javascript" id="inspectletjs">
window.__insp = window.__insp || [];
__insp.push(['wid', 1755897264]);
(function() {
function ldinsp(){if(typeof window.__inspld != "undefined") return; window.__inspld = 1; var insp = document.createElement('script'); insp.type = 'text/javascript'; insp.async = true; insp.id = "inspsync"; insp.src = ('https:'== document.location.protocol ?'https' :'http') + '://cdn.inspectlet.com/inspectlet.js'; var x = document.getElementsByTagName('script')[0]; x.parentNode.insertBefore(insp, x); };
setTimeout(ldinsp, 500); document.readyState != "complete" ?(window.attachEvent ? window.attachEvent('onload', ldinsp) : window.addEventListener('load', ldinsp, false)) : ldinsp();
})();
</script>
<!-- End Inspectlet Embed Code -->

クイックスタートガイド
=================

このページでは、DL4Jのexampleを動作させるために必要な事柄すべてをご説明します。

弊社の[Gitter Live Chat（Gitterライブチャット）](https://gitter.im/deeplearning4j/deeplearning4j)に参加されることをおすすめします。Gitterでは、ヘルプが必要な方へのサポートの提供やフィードバックの受付を行っております。なお、質問のある方は、以下のガイドにいくつかの質問に対する回答をご紹介しておりますので、こちらを先にお読みいただければ幸いです。ディープラーニングの初心者の方には[a road map for beginners（ディープラーニングの初心者ガイド）](../deeplearningforbeginners.html)、ディープラーニングに関するコースのサイト、読み物、その他のリソースもご紹介しています。

#### コードについて

Deeplearning4jはディープ・ニューラル・ネットワークを構成するドメイン固有の言語で、複数層で構成されています。すべては、これらの層やそれらのハイパーパラメータを組織化する`MultiLayerConfiguration`で開始します。

ハイパーパラメータとはニューラルネットワークがどのように学習するかを決定する変数です。ハイパーパラメータには、モデルの重みの更新回数、それらの重みの初期化方法、ノードに付与する活性化関数、使用すべき最適化アルゴリズム、モデルの学習速度に関する情報などが含まれています。以下は、設定の一例です。

``` java
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .iterations(1)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.05)
        // ... その他のハイパーパラメータ
        .backprop(true)
        .build();
```

Deeplearning4jでは、層を追加するには`NeuralNetConfiguration.Builder()`に`layer`を呼び、層の順序（下図のインデックスがゼロの層は入力層）におけるその位置、入力`nIn`と出力`nOut`それぞれのノード数、タイプを指定します。`DenseLayer`.

``` java
        .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
                .build())
```

いったんネットワークの設定が終わると、モデルを`model.fit`で訓練します。

## 必要なもの

* [Java （開発者バージョン）](#Java) 17、それ以降（**64ビットバージョンのみに対応しています。**）
* [Apache Maven](#Maven)
* [IntelliJ IDEA]（#IntelliJ）またはEclipse
* [Git](#Git)

この『クイックスタートガイド』の手順を踏むには、まず最初に上記のものがインストールされていなければなりません。DL4Jは、製品展開、自動構築ツールに精通したプロのJava開発者を対象としています。これらの分野で経験のある方ならDL4Jを使った作業は非常に簡単にできるでしょう。

Javaやこれらのツールの初心者の方々は、以下の詳細情報をお読みください。インスト―ルやセットアップについての情報を提供しております。そうでない方々は、**<a href="#examples">DL4Jのexamples</a>**にお進みください。

#### <a name="Java">Java</a>

Java 1.7、またはそれ以降のバージョンがない場合、現在の[Java Development Kit （JDK）をこちらから](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)ダウンロードしてください。互換性のあるJavaのバージョンを持っているかを調べるには、以下のコマンドを使用してください。

``` shell
java -version
```

64ビットのJavaがインストールされているかどうかを確認してください。32ビットのバージョンを使用した場合、エラーメッセージの`no jnind4j in java.library.path`が表示されます。

#### <a name="Maven">Apache Maven</a>

Mavenは、Javaのプロジェクトの依存関係を管理する自動化されたビルドツールです。IntelliJなどの統合開発環境（IDE）と連携しており、DL4Jのプロジェクトのライブラリを簡単にインストールすることができます。[Mavenの最新版のインストール、またはアップデート](https://maven.apache.org/download.cgi)を[指示](https://maven.apache.org/install.html)に従って行ってください。Mavenの最新版がインストールされているかどうかを調べるには、以下のコマンドを入力します。

``` shell
mvn --version
```

Macをお使いの方は、以下のコマンドを入力してください。

``` shell
brew install maven
```

MavenはJavaの開発者には広く使用されており、 DL4Jには必要不可欠です。これまでMavenを使う機会がなかった方は、[ApacheのMavenに関する概要](http://maven.apache.org/what-is-maven.html)、及びトラブルシューティングのヒントを載せた弊社の[Introduction to Maven for non-Java programmers（Javaのプログラマーでない方々のためのMaven初心者ガイド）](./maven)をお読みください。IvyやGradleなど[その他のビルドツール](../buildtools)も使用できますが、Mavenが最も使いやすいでしょう。

#### <a name="IntelliJ">IntelliJ IDEA</a>

統合開発環境（[IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)）を使うとAPI（アプリケーションプログラムインタフェース)を使ってニューラルネットワークをいくつかのステップを踏むだけで設定することができます。是非、[IntelliJ](https://www.jetbrains.com/idea/download/)を使用することをおすすめします。Mavenと連携して依存関係を処理することができるからです。[IntelliJのコミュニティ版](https://www.jetbrains.com/idea/download/)は無料です。

IDEといえば、他にも[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html)や[Netbeans](http://wiki.netbeans.org/MavenBestPractices)などが知られていますが、IntelliJの方がおすすめです。[Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)で分からないことなどを聞きたい場合も、IntelliJの方がより簡単に回答が得られます。

#### <a name="Git">Git</a>

[Gitの最新バージョン](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)をインストールしてください。すでにGitを使用している場合は、Gitを使って最新バージョンにアップデートすることができます。

``` shell
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

## <a name="examples">DL4Jのexamplesの簡単な作成手順</a>

1.コマンドラインを使用して、次のように入力します。

        $ git clone https://github.com/deeplearning4j/dl4j-examples.git
        $ cd dl4j-examples/
        $ mvn clean install

2.IntelliJを開き、「Import Project（プロジェクトをインポート）」を選んでください。次に、メインディレクトリの'dl4j-examples'を選んでください。

![select directory](../img/Install_IntJ_1.png)

3.'Import project from external model（外部モデルからプロジェクトをインポート）'を選び、Mavenが選択されているようにしてください。
![import project](../img/Install_IntJ_2.png)

4.ウィザードのオプションを続けます。`jdk`で始まるSDK（ソフトウェア開発キット）を選びます。（オプションを見えない場合は、プラス記号をクリックすると見れます。）そして、「Finish（完了）」をクリックします。IntelliJがすべての依存関係をダウンロードするのを待ちます。右下にある横線のバーが使えるようになっているのが見えます。

5.左側のファイルツリーから例をピックアップします。
![run IntelliJ example](../img/Install_IntJ_3.png)
ファイルを右クリックして作動させます。

## Using DL4J In Your Own Projects:POM.xmlファイルの設定

自分のプロジェクトでDL4Jを使用するには、Javaユーザは是非Maven、またはScala向けのツールのSBTなどを使用することをお勧めします。依存関係とそのバージョンの基本セットは以下の通りです。

- `deeplearning4j-core`：ニューラルネットワークの実装が含まれています。
- `nd4j-native`：DL4Jにパワーを提供するND4JのライブラリのCPUバージョン
- `canova-api` - Canovaは、弊社がベクトル化、ローディングを行うのに使っているライブラリです。

Mavenの各プロジェクトにはPOMファイルがあります。exampleを作動させると、POMファイルは、[こちら](https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml)のようになります。

IntelliJ内では、最初に実行するDeeplearning4jを選ぶ必要があります。`MLPLinearClassifier`がおすすめです。ネットワークがすぐに弊社のユーザー・インターフェースにある2つのデータグループを分類するのを確認できるからです。Githubにあるファイルは[こちら](https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java)からアクセスできます。

このexampleを実行するには、右クリックして、ドロップダウンメニューにある緑色のボタンを選択します。すると、IntelliJの下部のウインドウにスコアの連続が見えます。右端にある数字はネットワークの分類のためのエラースコアです。ネットワークが学習している場合は、時間の経過とともに各バッチが処理されていくにしたがってその数字は減少していきます。最後に、このウィンドウは、ニューラルネットワークのモデルがどのくらい正確になったかを報告します。

![run IntelliJ example](../img/mlp_classifier_results.png)

別のウィンドウでは、グラフによって、多層パーセプトロン（MLP）exampleのデータをどのように分類したかが表示されます。以下はその例です。

![run IntelliJ example](../img/mlp_classifier_viz.png)

お疲れ様でした！たった今、Deeplearning4jでの初めてのニューラルネットワークの訓練が完了しました。ほっと一息着いたところで、次のチュートリアルに進んでみませんか?[**MNIST for Beginners（初心者のためのMNIST）**](./mnist-for-beginners)では、画像の分類方法が学習できます。

## 次のステップ

1.Gitterに参加しましょう。Gitterには3つの大きなコミュニティチャンネルがあります。
  * [DL4J Live Chat（ライブチャット）](https://gitter.im/deeplearning4j/deeplearning4j)は、DL4Jのすべてのことについてを扱うメインチャンネルです。ほとんどの人々はこのチャットを使っています。
  * [Tunning Help](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp)は、ニューラルネットワークを始めた人々のために設けられています。初心者の方々は是非ご参加ください!
  * [Early Adopters](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters)は、弊社の次のリリースのチェックや改善のお手伝いをしてくださっている方々向けです。注意：このコミュニティーは経験者向けです。
2.[Introduction to deep neural networks（ディープニューラルネットワークについて）](ja-neuralnet-overview)または[弊社の詳細チュートリアルの一つ](../tutorials)をお読みください。
3.より詳細の[Comprehensive Setup Guide（セットアップ全ガイド）](ja-gettingstarted)をお読みください。
4.[DL4Jのガイド集](./documentation)をご覧ください。

### その他のリンク

- [Deeplearning4j artifacts on Maven Central（Maven CentralにあるDeeplearning4jのアーチファクト）](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
- [ND4J artifacts on Maven Central（Maven CentralにあるND4Jのアーチファクト）](http://search.maven.org/#search%7Cga%7C1%7Cnd4j)
- [Canova artifacts on Maven Central（Maven CentralにあるCanovaのアーチファクト）](http://search.maven.org/#search%7Cga%7C1%7Ccanova)

### トラブルシューティング

**質問：**ウインドウズで64ビットのJavaを使用しているのですが、いまだにエラーメッセージの`no jnind4j in java.library.path`が表示されます。

**回答：**パスに互換性のないDLLがある可能性があります。これらを無視させるには、以下をVMパラメーターとして追加してください。（Run -> Edit Configurations -> IntelliJのVMオプション）

```
-Djava.library.path=""
```

**質問：**次のようなエラーが発生します。`Intel MKL FATAL ERROR:Cannot load mkl_intel_thread.dll`.そしてJVMがシャットダウンしてしまいます。（クラッシュはしませんが、ストップしてしまいます ... ）

**回答：**`rc3.10`やそれ以降（弊社では0.4.0）は、ライブラリ`libnd4j`がパスにあってもIntelのマス カーネル ライブラリー（MKL）を正常に読み込みません。しかし、この問題は、`System.loadLibrary("mkl_rt")`を追加すると解消されます。
