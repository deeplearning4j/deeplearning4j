---
title:
layout: ja-default
---

# <a name="quickstart"> クイックスタート

## 必要なもの
このガイドは以下のものが予め利用できることを想定しています。

- Java
- IDE(例: IntelliJ IDEA)
- Maven(Javaのビルドツール)
- [Canova](https://github.com/deeplearning4j/Canova)(機械学習用のベクトル化ツール)
- GitHub(optional)

上記のものを新たにインストールする必要があれば、[Getting Started Guide](http://nd4j.org/ja-getstarted.html)をご参照ください。[ND4J](https://github.com/deeplearning4j/nd4j)はDL4J用の線形代数ライブラリです。
上記にあげたものだけで基本的には十分で他のものをインストールする必要はありません。

## 利用までのステップ
1. [nd4j](https://github.com/deeplearning4j/nd4j/), [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/), [Canova](https://github.com/deeplearning4j/Canova), [example](https://github.com/deeplearning4j/dl4j-0.4-examples)プロジェクトをダウンロードしてくる。(git cloneを用いる)
2. それぞれのプロジェクトに対して`mvn clean install -DskipTests -Dmaven.javadoc.skip=true`でビルドします。
3. IntelliJ IDEAのようなIDEでMavenプロジェクトとして上記のexampleをインポートする。
4. デフォルトのバックエンドは`nd4j-jblas`に設定されています。(Windowsの場合は`nd4j-java`に変更することを推奨しています)
4. ソースツリーからDBNSmallMnistExample.javaを選択。
5. 上記のクラスを実行する。

Irisのような小さなデータセットではF1スコアを約0.55にすることを推奨しています。

## 依存関係

バックエンドとはDL4Jのニューラルネットワークが利用する線形代数ライブラリの処理基盤です。バックエンドはマシンのチップに依存します。CPUはJblas、Netlib Blasで高速に、
GPUではJcublasで高速に動作します。依存しているライブラリが何か分かっている場合はMaven Centralで探して”Latest Version”をク リックしてください。記載されているxmlの断片をあなたのプロジェクトのルート直下にある pom.xmlにコピーアンドペーストしてください。BLASのバックエンドに関しては以下のようになるはずです。

    <dependency>
      <groupId>org.nd4j</groupId>
	  <artifactId>nd4j-java</artifactId>
	  <version>${nd4j.version}</version>
    </dependency>

`nd4j-java`はWindowsでのセットアップを楽にするためBlasを要求しません。exampleのDBNs, deep-belief netsのプロジェクトで動作しますがその他では動作しません。


    <dependency>
      <groupId>org.nd4j</groupId>
	  <artifactId>nd4j-jblas</artifactId>
	  <version>${nd4j.version}</version>
    </dependency>


`nd4j-jblas`はすべてのexampleで動作します。Windowsでこれをインストールするには[Getting Started](http://deeplearning4j.org/gettingstarted.html)を参照ください。

## AWS上でコマンドラインを利用する場合
AWS上にDL4Jをインストールした場合、IDEではなくコマンドラインで実行をする必要があります。この場合はインストールに関しては上記に記載された通りに行ってください。
exampleを走らせるためのコマンドはversionなどによっても異なりますが下記がテンプレートとなります

```
$ java -cp target/nameofjar.jar fully.qualified.class.name
```

例えば具体的には下記のようになります。
```
$ java -cp target/deeplearning4j-examples-0.4-SNAPSHOT.jar org.deeplearning4j.MLPBackpropIrisExample
```

つまりversionを更新したり、走らせるクラスが異なる場合は下記のワイルドカードを変更していくことになります。

```
$ java -cp target/*.jar org.deeplearning4j.*
```

exampleに変更を加える場合(例えばsrc/main/java/org/deeplearning4j/multilayerの中のMLPBackpropIrisExample)は再ビルドが必要です。


## ソースからのインストール
注意:Mavenにあるパッケージを使うだけであればソースからインストールする必要はありませ ん。
1. MavenをダウンロードしPATHに通します。
2. Deeplearning4jのプロジェクトにあるsetup.shあるいはWindowsの場合はsetup.batを実行しま
す。(この実行ファイルをダウンロードするにはGitHubアカウントとgitが必要ですのであらか じめ設定をしておいてください。すでにダウンロードしてある場合にはgit pullのみで構いませ ん。)
