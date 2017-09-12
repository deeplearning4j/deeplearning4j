- - -
title:Configuring Automated Build Tools
layout: default
---

## 自動ビルドツールの設定

Deeplearning4j、ND4J、DataVecのユーザーの方々にはMavenの使用をお勧めしていますが、Ivy、Gradle、SBTその他のツールのビルドファイルの設定方法の説明もあると役に立つでしょう。というのも、例えばGoogleはAndroidプロジェクトについてはMavenでなくGradleを選択する、ということなどもあるからです。 

以下の手順は、deeplearning4j-api、deeplearning4j-scaleout、ND4Jのバックエンドなど、DL4JすべてのND4Jのサブモジュールに適用されます。すべてのプロジェクトの**最新バージョン**またはサブモジュールは[Maven Central](https://search.maven.org/)で見つかります。2017年1月以降は、最新バージョンは`0.7.2`となっています。ソースからの構築の場合は、`0.7.3-SNAPSHOT`が最新バージョンです。

## Maven

Deeplearning4jをMavenで使用するには、POM.xmlに以下を追加してください。

    <dependencies>
      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>deeplearning4j-core</artifactId>
          <version>${弊社のexamplesがある http://github.com/deeplearning4j/dl4j-examples からバージョンを探してください}</version>
      </dependency>
    </dependencies>

正常なMavenの設定で動作するexampleは、deeplearning4Jはnd4JとDataVecと依存関係を持ちますので注意してください。弊社のexamplesについては、[こちら](http://github.com/deeplearning4j/dl4j-examples)をご覧ください。

## Ivy

lombokをivyで使用するには、ivy.xmlに以下を追加してください。

    <dependency org="org.deeplearning4j" name="deeplearning4j-core" rev="${弊社のexamplesがある http://github.com/deeplearning4j/dl4j-examples からバージョンを探してください}" conf="build" />

## SBT

Deeplearning4jをSBTで使用するには、build.sbtに以下を追加してください。

    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "${弊社のexamplesがある http://github.com/deeplearning4j/dl4j-examples からバージョンを探してください}"

## Gradle

Deeplearning4jをGradleで使用するには、依存関係ブロックにあるbuild.gradleに以下を追加してください。

    provided "org.deeplearning4j:deeplearning4j-core:${弊社のexamplesがある http://github.com/deeplearning4j/dl4j-examples からバージョンを探してください}"

## Leiningen

Clojureのプログラマーの方々の中には、Mavenで使用できるように、[Leiningen](https://github.com/technomancy/leiningen/)や[Boot](http://boot-clj.com/)を使用したい方もおられることでしょう。Leiningenのチュートリアルが[こちら](https://github.com/technomancy/leiningen/blob/master/doc/TUTORIAL.md)にありますので、参考にしてみてください。

注意: Eclipseインストレーションにインストールするには、ND4J、DataVec、Deeplearning4jをダウンロードするか、Maven/Ivy/GradleからダウンロードされたそれらのJARファイルをダブルクリックする必要があります。

## バックエンド

[バックエンド](http://nd4j.org/backend) や他の[依存関係](http://nd4j.org/dependencies)はND4Jのウェブサイトで説明しておりますのでご覧ください。
