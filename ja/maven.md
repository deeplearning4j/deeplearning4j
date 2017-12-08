---
title:Maven for Python Programmers
layout: default
---

#  PythonのプログラマーのためのMavenガイド

[Maven](https://ja.wikipedia.org/wiki/Apache_Maven)は、Javaのプログラマーが最もよく使用するビルド自動化ツールです。Mavenの特徴それぞれにすべてマッチするPythonのツールはありませんが、Pythonの[Pip](https://ja.wikipedia.org/wiki/Pip)やPyBuilder、[Distutils](http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html)などのパッケージ管理システムに似ています。 

また、MavenはPythonのプログラマーには不気味なほど構文がお馴染みの[ScalaのAPI](http://nd4j.org/scala.html)を提供するDeeplearning4jを使用するには唯一最も便利な方法であると同時に、他の強力な機能も利用できます。 

ビルド自動化ツールとして、Mavenはソースコードからバイトコードをコンパイルし、オブジェクトファイルを実行可能ファイルやライブラリファイルにリンクさせます。その成果物は、Javaソースと展開用リソースから作成されたJARファイルです。 

（[JAR](https://ja.wikipedia.org/wiki/JAR_(%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%83%95%E3%82%A9%E3%83%BC%E3%83%9E%E3%83%83%E3%83%88))とは **Java ARchive** のことで、数多くのJavaクラスファイル、関連するメタデータ、テキストや画像などのリソースを集計するパッケージファイルのフォーマットです。これは圧縮ファイルフォーマットで、Javaランタイムがクラスのセットとそのリソースを展開するのを助けます。） 

Mavenは動的にJavaライブラリとMavenプラグインをMavenの中央リポジトリからダウンロードします。これらはPOM.xmlにあり、プロジェクト・オブジェクト・モデルを保管するXMLファイル内で指定されています。 

![Alt text](./img/maven_schema.png)

*Maven:The Complete Reference* から引用 

		コマンドラインからmvn installを実行すると、リソースの処理、ソースのコンパイル、単体テストの実行、JARの作成、他のプロジェクトで再使用するためのJARのローカルリポジトリへのインストールが行われます。 

Deeplearning4jのように、Mavenは設定より規約に依存しています。つまりプログラマーが各パラメータをそれぞれの新しいプロジェクトに指定しなくても実行できるようにする既定値が提供されます。 

IntelliJとMavenの両方をインストールしていれば、IntelliJはIDEで新しいプロジェクトを作成しているときにMavenを選択するのを許可し、ウイザードに案内します（このプロセスの詳細は[弊社のクイックスタートガイド](https://deeplearning4j.org/ja/quickstart)をお読みください）。つまり、他のどこにも行かずにIntelliJ内からビルドを行うことができるのです。 

また、別の方法では、新しくインストールするためにコマンドプロンプトにある自分のプロジェクトのルートディレクトリからMavenを使用することもできます。

		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		
上記のコマンドは、インストールを実行する前に、すべてのディレクトリにあるコンパイルされたファイルを削除することをMavenに指示するものです。これにより、確実にビルドが完全に新しいものであることを確認します。


これまでにApache Mavenについての有用な本が何冊か書かれています。これらの書籍は、オープンソースプロジェクトを支援する会社であるSonatypeのウェブサイトから入手可能になっています。 

### Mavenのトラブルシューティング

* 3.0.4などのMavenの以前のバージョンは、NoSuchMethodErrorなどの例外を投げることが多いです。このような問題は、Mavenの最新バージョンにアップグレードすると解消されます。 
* Mavenをインストールした後、「mvnは、内部コマンドまたは外部コマンド、操作可能なプログラムまたはバッチファイルとして認識されませんでした。（mvn is not recognised as an internal or external command, operable program or batch file.）」と書かれたメッセージが表示されることがあります。これは、[PATH変数](https://www.java.com/en/download/help/path.xml)にMavenが必要であるということを意味します。 
* DL4Jのコードベースが大きくなればなるほど、ソースからのインストールにはさらに多くのメモリーが必要になります。DL4J構築中に「Permgen error」が発生したら、さらにヒープ領域を増やなければならないかもしれません。それをするには、隠しファイル「bash_profile」を見つけて修正し、環境変数をbashに追加する必要があります。これらの変数を見るには、コマンドラインに`env`と入力してください。さらにヒープ領域を増やすには、コンソールにコマンドを
      「echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile」と入力してください。

### その他の情報

* [Maven by Example](https://books.sonatype.com/mvnex-book/reference/public-book.html)（サンプルを使ったMavenの解説）
* [Maven:The Complete Reference](https://books.sonatype.com/mvnref-book/reference/public-book.html)（Mavenの完全ガイド）
* [Developing with Eclipse and Maven](https://books.sonatype.com/m2eclipse-book/reference/)（EclipseとMaveを使った開発）
