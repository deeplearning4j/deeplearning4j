---
title: Deeplearning4jのフルインストール
layout: ja-default
---

# フルインストール

このインストールは複数の段階の手順に従って行います。質問やコメント等は、是非、[Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)でお聞かせください。弊社のスタッフがお手伝いいたします。恥ずかしがる必要は全くありません。いつでも気軽にご連絡ください。また、deep-learningの初心者の方には、[deep-learningの初心者ガイド](../deeplearningforbeginners.html)もご用意いたしました。 

exampleを簡単なステップで走らせるには、[クイックスタート](../quickstart.html)をお読みください。また、もしクイックスタートをまだ読まれていない方は、以下の説明をおお読みになる前に、是非そちらを読まれることをお勧めします。DL4Jを始めるごく簡単な方法をご紹介しているからです。 

Deeplearning4jのインストールに必要なものは、[ND4Jを「はじめましょう」](http://nd4j.org/getstarted.html)でご紹介しています。ND4Jとは、DL4Jのニューラルネットワークが使用する線形代数の計算エンジンです。

1. [Java 7、またはそれ以降のバージョン](http://nd4j.org/getstarted.html#java) 
2. [統合開発環境：IntelliJ](http://nd4j.org/getstarted.html#ide-for-java) 
3. [Maven](http://nd4j.org/getstarted.html#maven)

上記をインストールした後、以下をお読みください。

6. OS関連の説明ガイド
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
8. [GitHub](http://nd4j.org/getstarted.html#github)
9. <a href="#eclipse">Eclipse</a>
10. <a href="#trouble">トラブルシューティング</a>
11. <a href="#results">再現可能な結果</a>
12. <a href="#next">次のステップ</a>

### <a name="linux">Linux</a>

* Deeplearning4jは、CPUに対応したBlasの様々な形態に依存しているため、 Blasへのネイティブバインディングが必要です。

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

OpenBlasに関する情報については、[こちらのセクション](#open)をお読みください。

* GPUが壊れている場合は、コマンドを入力する必要があります。まず、Cudaがどこにインストールされるかを確認してください。以下のようになっているはずです。

         /usr/local/cuda/lib64

それから、ターミナルに*ldconfig*と入力し、続けてCudaへリンクされるファイルパスを入力します。つまり、コマンドは下記のようなものになります。

         ldconfig /usr/local/cuda/lib64

それでもJcublasをロードできなければ、パラメータの-Dをコードに追加してください（JVM引数です）。

         java.library.path (settable via -Djava.librarypath=...) 
         // ^ for a writable directory, then 
         -D appended directly to "<OTHER ARGS>" 

統合開発環境にIntelliJを使用している場合、既にこれは動作しているはずです。 

### <a name="osx">OSX</a>

* Blasは既にOSXにインストールされています。  

### <a name="windows">Windows</a>

* Windowsでのインストールは、常に簡単というわけではありません。しかし、Deeplearning4のように、オープンソースのdeep-learningプロジェクトで、実際にWindows利用者向けのサポートに熱心なものは、数少ないのが現状です。詳細については、弊社のND4Jページにある[Windows用のセクション](http://nd4j.org/getstarted.html#windows)をお読みください。 

* お使いのコンピューターが64ビットでも[MinGW 32 bits](http://www.mingw.org/)をインストール（ダウンロードボタンは、右上に表示）し、[Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)をダウンロードします。 

* [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)をインストールします。（Intelコンパイラーがあるか質問が表示されますが、これはないはずです。）

* Lapackは、[VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)の代替を提供します。[Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)についての解説も読んでおくと有用でしょう。 

* あるいは、MinGWでなく、Blas dllファイルをパスの中のフォルダーにコピーすることもできます。例えば、 MinGWのbinフォルダへのパスは、/usr/x86_64-w64-mingw32/sys-root/mingw/binです。Windowsのパスの変数についての詳細は、[こちらのStackOverflowサイトの上部にある回答](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)をお読みください。 

* Cygwinには対応していません。**DOS Windows**からDL4Jをインストールする必要があります。  

* ファイルの[WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100)を実行すると、Windowsのインストールでのデバッグができます。その正しい出力例は[こちら](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f)をご覧ください。最初にダウンロードし、コマンドウィンドウ／ターミナルを開きます。ダウンロードされたディレクトリに`cd`コマンドを入れます。`WindowsInfo`と入力し、Enterキーを押します。この出力をコピーするには、コマンドウィンドウ上で右クリックし、 [select all（すべて選択）]を選び、Enterキーを押します。すると、クリップボードに出力されます。

**Windows**向けのOpenBlas（以下を参照）は、[こちらのファイル](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1)をダウンロードしてください。`C:/BLAS`など、どこか場所を決めて解凍してください。このディレクトリを、システムの環境変数`PATH`に追加します。

### <a id="open"> OpenBlas </a>

x86のバックエンドにあるライブラリーが動作することを確認するには、システムパスの`/opt/OpenBLAS/lib`が必要です。その後で、以下のコマンドをプロンプトに入力してください。

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

これは、[Spark](http://deeplearning4j.org/spark)がOpenBlasで使用できるようにするためです。

OpenBlasが正しく動作していない場合は、次の手順に従ってください。

* Openblasがインストール済みであれば、削除します。
* `sudo apt-get remove libopenblas-base`を実行します。
* OpenBLASの開発版をダウンロードします。
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* **Linux**の場合、`libblas.so.3`や`liblapack.so.3`のシンボリックリンクが`LD_LIBRARY_PATH`のどこでも存在することを再度確認してください。もし存在しなければ、`/usr/lib`へリンクを追加します。以下のように設定することができます（-sを入れるとリンクがシンボリックになります）。 

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"

* 「from-here」は、まだ存在しないシンボリックリンクを作成したものです。StackOverflowのシンボリックリンク作成方法ガイドがありますので、[こちら](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux)をお読みください。「Linux man page」は、[こちら](http://linux.die.net/man/1/ln)をお読みください。
* 最後に統合開発環境を再起動します。 
* ネイティブのBlasを **CentOS 6**で作動させるための詳細は、[CentOS 6](https://gist.github.com/jarutis/912e2a4693accee42a94)または [CentOS 6.5](https://gist.github.com/sato-cloudian/a42892c4235e82c27d0d)をお読みください。

**Ubuntu** (15.10)のOpenBlasについての説明ガイドは[こちら](http://pastebin.com/F0Rv2uEk)をお読みください。

### <a name="eclipse">Eclipse</a> 

`git clone`を実行してから、以下のコマンドを入力してください。

      mvn eclipse:eclipse 
  
これにより、ソースがインポートされ、すべてがセットアップされます。 

弊社は、何年にも渡りEclipseを使用しましたが、Eclipseと似たインターフェイスを持つIntelliJをお勧めします。Eclipseのモノリス型アーキテクチャだと、他社のものでも弊社のコードでも奇妙なエラーが発生することが度々あるからです。 

Eclipseを使う場合は、[Lombok plugin](https://projectlombok.org/)をインストールする必要があります。また、[Eclipse用のMavenプラグイン、eclipse.org/m2e](https://eclipse.org/m2e/)も必要になります。

Michael Depies氏が、 [EclipseでのDeeplearning4jのインストールガイド](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/)を作成し、提供していますので、ご参照ください。

### <a name="trouble">トラブルシューティング</a>

* エラーメッセージについて質問があれば、[Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j)にお気軽にご連絡ください。また、質問の際には、以下の情報を準備しておいてください。（前もって準備していただきますと、より素早くご質問に対処できます。) 

      * オペレーティング・システム（Windows、OSX、Linux）とそのバージョン 
      * Javaバージョン（7、8） : ターミナル/コマンドプロンプトにjava -versionと入力すると分かります。 
      * Maven のバージョン : ターミナル/コマンドプロンプトにmvn --versionと入力すると分かります。
      * スタックトレース:gistのエラーコードをペーストし、リンクをお送りください。[https://gist.github.com/](https://gist.github.com/)
* 既にDL4Jがインストールされており、exampleがエラーを多く送出させている場合、ライブラリをアップデートしてください。Mavenについては、[Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)を使ってPOM.xmlファイルにあるバージョンを最新のものにアップデートしてください。ソースについては、3つのディレクトリである[ND4J](http://nd4j.org/getstarted.html)、Canova、DL4Jをこの順で`git clone`、次に`mvn clean install -Dskiptests=true -Dmaven.javadoc.skip=true`と実行してください。
* exampleを実行するとき、[F1スコア](../glossary.html#f1)結果が低くなるかもしれません。F1スコアとは、ネットの分類作業の精確さを示すものです。しかし、この場合、 F1値が低い原因は、精確性が低いからではなく、小さいデータセットを使用しているためです。小さめのデータセットだと素早く走らせることができるからです。小さめのデータセットは、大きいものより代表的にはなりませんが、結果は様々に異なります。例えば、非常に小さいexampleのデータでは、弊社のdeep-belief networkのF1スコアは0.32から1.0にまで及びます。 
* Deeplearning4jには**オートコンプリート機能**が含まれます。どのコマンドが使用可能か分からないときは、任意の文字を1つ打つと、ドロップダウンメニューが下記のように
![Alt text](../img/dl4j_autocomplete.png)表示されます。
* すべてのDeeplearning4jのクラスとメソッドのための**Javadoc**は、[こちら](http://deeplearning4j.org/doc/)です。
* コードベースが大きくなればなるほど、さらにメモリー量が必要になります。DL4J構築中に`Permgen error`が発生したら、さらに**ヒープ領域**を増やす必要があるかもしれません。これには、隠しファイル`.bash_profile`を見つけて修正し、環境変数がbashに設定される必要があります。さらに変数を見るには、コマンドラインに`env`と入力してください。さらにヒープ領域を増やすには、コンソールに次のコマンドを入力してください。
      「echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile」
* 3.0.4などのMavenの以前のバージョンは、NoSuchMethodErrorなどの例外を送出する傾向があります。このようなトラブルは、Mavenの最新バージョン（現在は3.3.x）にアップグレードすることによって解消されます。お使いのMavenバージョンを調べるには、コマンドラインに`mvn -v`と入力してください。
* Mavenをインストールした後、`mvn is not recognised as an internal or external command, operable program or batch file.`と書かれたメッセージが表示される可能性があります。これは、他の環境変数と同様に変更可能な[PATH変数](https://www.java.com/en/download/help/path.xml)にMavenが必要であるということを意味します。  
* エラーの`Invalid JDK version in profile 'java8-and-higher':Unbounded range:[1.8, for project com.github.jai-imageio:jai-imageio-core com.github.jai-imageio:jai-imageio-core:jar:1.3.0`が見つかった場合、Maven関連の問題が発生している可能性がありますので、Mavenのバージョンを3.3.xにアップデートしてください。
* ND4Jの依存関係をコンパイルするには、CとC++の**開発ツール**をインストールする必要があります。[弊社のND4Jガイドをお読みください。](http://nd4j.org/getstarted.html#devtools)
* [Java CPP](https://github.com/bytedeco/javacpp)のinclude pathは、常に**Windows**で動作するとは限りません。これを解決する1つの方法はVisual Studioのincludeディレクトリからヘッダーファイルを取り、それらをJavaがインストールされたJava Run-Time Environment (JRE)のincludeディレクトリに入れることです。これによりstandardio.hなどのファイルに影響が出ます。詳細については[こちら](http://nd4j.org/getstarted.html#windows)をお読みください。 
* GPUのモニターに関する説明ガイドは、[こちら](http://nd4j.org/getstarted.html#gpu)をお読みください。
* Javaを使用する主な理由の一つは、 **[JVisualVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jvisualvm.html)**に既に診断機能があるからです。Javaがインストールされていれば、コマンドラインに`jvisualvm`と入れさえすると、 CPU、ヒープ、PermGen、クラス、スレッドのビジュアル情報を見ることができます。例えば、以下の例をご覧ください。右上の`サンプラ`タブをクリックし、CPU、またはメモリーボタンを押します。すると、ビジュアル情報が得られます。 
![Alt text](../img/jvisualvm.png)
* DL4Jを使用している際に発生する問題の原因は、ユーザーが機械学習に関する情報やテクニックに十分精通していないためである場合もあります。 Deeplearning4jのユーザー皆さんに、このサイト以外のリソースも利用して基礎を学習することをお勧めします。機械学習やdeep-learningの学習リソースのリストを準備しましたので、[こちら](../deeplearningpapers.html)をお読みください。弊社はDL4Jを一部ドキュメント化しましたが、deep-learning用に使用するには、コードの一部は生で、ドメイン固有言語のままです。
* **Clojure**アプリケーションから`deeplearning4j-nlp`を使い、Leiningenでuberjarを構築するときは、akkaの`reference.conf`リソースファイルが適切にマージされるよう、`project.clj`に、`:uberjar-merge-with {#"\.properties$" [slurp str spit] "reference.conf" [slurp str spit]}`と指定してください。（.propertiesファイルのマップへの最初の入力は、通常のデフォルトであることにご注意ください）。この設定が行われていない場合、結果のuberjarから実行しようとすると、次のような例外が送出されます。`Exception in thread "main" com.typesafe.config.ConfigException$Missing:No configuration setting found for key 'akka.version'`
* OSXの浮動小数点のサポートにはバグが多くあります。examplsの実行でNaNが数多く表示される場合、データのタイプを`double`に切り替えてください。
* Java 7のfork-joinにはバグがありますが、Java 8にアップデートすることにより修正されます。以下のようなOutofMemoryエラーが発生する場合は、fork-joinに問題があります。`java.util.concurrent.ExecutionException: java.lang.OutOfMemoryError`
....`java.util.concurrent.ForkJoinTask.getThrowableException(ForkJoinTask.java:536)`

### <a name="results">再現可能な結果</a>

ニューラルネットの重みはランダムに初期化されます。つまり、モデルは、毎回、重み空間の異なる位置から学習を開始し、これにより局所的最適が変わります。再現性のある結果を求めるユーザーは、同じランダムの重みを使用する必要がありますが、モデルが作成される前に初期化する必要があります。以下のコマンドラインを使うと、同じランダムな重みで、再度、初期化することができます。

      Nd4j.getRandom().setSeed(123);

### <a name="next">次のステップ:IRISのexampleとニューラルネットワークの構築</a>

ニューラルネットワークの構築を開始するには、[ニューラルネットワークの概要](http://deeplearning4j.org/neuralnet-overview.html)にて詳細をお読みください。

素早く走らせるには[IRISのチュートリアル](../iris-flower-dataset-tutorial.html)をお読みください。 *deep-belief network*の基本的なメカニズムを理解するには、[制限付きボルツマン・マシン](../restrictedboltzmannmachine.html)をお読みください。

新しいプロジェクトを開始して必要な[POMの依存関係](http://nd4j.org/dependencies.html)を入れるには、[ND4Jをはじめましょう](http://nd4j.org/getstarted.html)をお読みください。 
