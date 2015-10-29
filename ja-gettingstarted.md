---
layout: ja-default
---

# Deeplearning4jをはじめましょう

コンテンツ ("日本語サイトは準備中です。英語サイトをご覧ください"。[English version](../gettingstarted.html))

* <a href="#quickstart">クイックスタート</a>
* <a href="#all">Deeplearning4jのインストール方法(All OS)</a>
    * <a href="#ide-for-java">IDE</a>
    * <a href="#maven">Maven</a>
    * <a href="#github">Github</a>
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
* <a href="#source">ソースの取得方法</a>
* <a href="#eclipse">Eclipse</a>
* <a href="#trouble">トラブルシューティング</a>
* <a href="#next">Next Steps</a>

## <a name="quickstart">クイックスタート</a>

[クイックスタート](../ja-quickstart.html)の項目では、どのようにdeeplearning4jをスタートするか紹介しております。

## <a name="all">インストール方法: All OS</a>

DeepLearning4Jを実行するためには[Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) もしくはaboveが必要となります。Java7のダウンロード方法に関しては[ND4JのGet started](http://nd4j.org/ja-getstarted)の項目をご参照ください。

## ND4J: Numpy for the JVM

[ND4J](http://nd4j.org/ja-getstarted) は、Javaを基本にしたコンピューターエンジンです。また、DL4Jを実行するためには、ND4Jをダウンロードする必要があります。

## <a id="ide-for-java">Integrated Development Environment</a>

### Integrated Development Environmentとは

Integrated Development Environment[IDE](https://ja.wikipedia.org/wiki/%E7%B5%B1%E5%90%88%E9%96%8B%E7%99%BA%E7%92%B0%E5%A2%83)とは、ソフトウェアの開発において用いられるエディタ、コンパイラ、リンカ、デバッガ、その他の支援ツールなどを統合・統一化した開発環境のことを指します。IDEには、ソフトウェア開発に必要な最低限のツールがすべて含まれているため、これを導入することで、インストールしたMavenとGitHubの操作を統一して行うことができます。 

### なぜIDEが必要か

IDSを活用することで、コードを入力するだけで簡単にシステムをセットアップができるようになります。IDEは一般的にMavenとセットで使われるため、Mavenのダウンロードをおすすめしております。

### イントール状況の確認

インストールプログラムをご確認ください。

### インストール方法

[intellij](https://www.jetbrains.com/idea/download/)のfree community editionをお勧めいたします。

以下のIDEも同様にご活用いただけます。

[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) or [Netbeans](http://wiki.netbeans.org/MavenBestPractices).

インストール後、以下のサイトからND4Jプロジェクトをダウンロードいただけます。

[Intellijの場合](http://stackoverflow.com/questions/1051640/correct-way-to-add-lib-jar-to-an-intellij-idea-project)、
[Eclipseの場合](http://stackoverflow.com/questions/3280353/how-to-import-a-jar-in-eclipse) 、 [Netbeansの場合](http://gpraveenkumar.wordpress.com/2009/06/17/abc-to-import-a-jar-file-in-netbeans-6-5/)

## <a id="maven">Maven</a>

### Mavenとは
 MavenとはJava用プロジェクト管理ツールです。([Mavenホームページ](http://maven.apache.org/what-is-maven.html)) Mavenをインストールすることで、最新版のND4Jの[JAR](http://ja.wikipedia.org/wiki/JAR_%28%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%83%95%E3%82%A9%E3%83%BC%E3%83%9E%E3%83%83%E3%83%88%29)を自動的にアップロードし続けることができます。

### なぜ必要か
 Mavenを活用することで、より簡単にND4JとDeeplearning4j projectsをインストールすることができます。なお、最終的にダウンロードする[IDE](http://ja.wikipedia.org/wiki/%E7%B5%B1%E5%90%88%E9%96%8B%E7%99%BA%E7%92%B0%E5%A2%83)を操作するうえでも、Mavenは役立ちます。IDEまた、もしMavenの内容をご理解いただいている方は、[当社ホームページ](http://deeplearning4j.org/downloads.html) ページにアクセスいただくことで、この過程をスキップすることができます。

### イントール状況の確認
コマンドラインに、以下のコードをご入力ください。

		mvn --version

### インストール方法
[Mavenホームページ](https://maven.apache.org/download.cgi)を通じて、無料でダウンロードいただけます

![Alt text](../img/maven_downloads.png) 

ページの下部にある、お使いのOperating Systemごとの説明に沿って、インストールを進めてください。
 “Unix-based Operating Systems (Linux, Solaris and Mac OS X).”はこのような形で表示されております。
 
![Alt text](../img/maven_OS_instructions.png) 

ここまでの作業を完了すると、IDEを使って新しいプロジェクトを作ることができます。

![Alt text](../img/new_maven_project.png) 

IntelliJのWindowを通じて、下に表示されている画面が表示されます。まずはじめに名前を入力します。

![Alt text](../img/maven2.png) 

 "Next"を押していただくと、次のウィンドウが表示されますので、"Deeplearning4j"と名前を入力してください。
 
 ![Alt text](../img/maven4.png) 
 
 これでIntelliJのpom.xml fileにアクセスでき、以下のよう表示されます。
 
 ![Alt text](../img/pom_before.png) 
 
 次に<dependencies>セクションに[dependency](https://github.com/SkymindIO/deeplearning4j/tree/0.0.3.3)を加えていく必要があります。これはCPUsやGPUsによって異なりますので、それぞれに適応する形で"nd4j-api"と a linear-algebra backend like "[nd4j-jblas](http://search.maven.org/#search%7Cga%7C1%7Cnd4j-jblas)" か"nd4j-jcublas"を選択してください。これらはすべて <a href="http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j-core">こちら</a>から取得できます。 
 
 ![Alt text](../img/search_maven_latest_version.png) 
 
 "latest version" を選択し、コピーを行ってください。
 
 ![Alt text](../img/latest_version_dependency.png) 
 
 コピーした内容を<dependencies>セクションにペーストすると、以下の表示内容になります。
 
 ![Alt text](../img/pom_after.png) 
 
 これで設定は完了になります。これ以降はIntelliJに新たなファイルを作ることも、 ND4Jの APIを利用することも可能になります。
 もし新たなアイデアが必要な場合には、[intro](http://nd4j.org/introduction.html)をご覧ください。

# <a id="github">GitHub</a>

## GitHubとは

[Github](https://ja.wikipedia.org/wiki/GitHub) は [Revision Control System](http://ja.wikipedia.org/wiki/Revision_Control_System)に基づいた、ソフトウェア開発プロジェクトのための共有ウェブサービスであり, [open source](http://ja.wikipedia.org/wiki/オープンソース) projects向けの無料アカウントを提供しています。

### なぜ必要か
 GitHubはこのシステムを使う上で必ずしも必要なものではありません。しかし、ND4Jファイルのダウンロードやプロジェクトの状況、バグの報告をチームメンバー間で共有する際には、GitHubが役立ちます。

### イントール状況の確認
 インストールプログラムにて、ご確認いただけます。

### インストール方法
 以下のURLを通じて無料でダウンロードいただけます。
 
[Macはこちら](https://mac.github.com/), [Windowsはこちら](https://windows.github.com/)

ND4Jのファイルを複製するためには以下の文章をterminal (Mac) もしくは Git Shell (Windows)へ入力してください。

      git clone https://github.com/SkymindIO/nd4j

## <a name="linux">Linux</a>

*正常にインストールを完了するためには、Blasの初期設定が必要になります。

        Fedora/RHEL
        yum -y install blas
        
        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

* もしGPUが壊れていた場合、コマンド入力が必要になります。まずはじめに、下記のコマンドを入力し、Cudaがどこにインストールされているかを確認してください。

         /usr/local/cuda/lib64

次に*ldconfig* を調べていただく際のコマンドは、以下の通りになります。

         ldconfig /usr/local/cuda/lib64

もしJcublasがインストールできない場合には、以下のようにparameter -D をコードに入力いただく必要があります。 (これは一種のJVM になります。):

         java.library.path (settable via -Djava.librarypath=...) 
         // ^ for a writable directory, then 
         -D appended directly to "<OTHER ARGS>" 

もしIDEとしてIntelliJをお使いの場合には、既にこの設定は完了しております。

## <a name="osx">OSX</a>

* JblasはすでにOSXに対応しております。

## <a name="windows">Windows</a>

* [Maven](http://maven.apache.org/download.cgi)のダウンロードページで、どのようにwindows環境下でJavaとMavenをダウンロードすれば良いかという説明がされております。 この設定を完了するためには、[environment　variables](http://www.computerhope.com/issues/ch000549.htm)が適切な環境である必要があります。 

* [Anaconda](http://docs.continuum.io/anaconda/install.html#windows-install)をインストールしてください。もしシステムが64-bit ない場合は、同じページにある32-bitをダウンロードしてください。 (Deeplearning4jはAnacondaを通じてmatplotlibを活用します。) 

* [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)をダウンロードしてください。(Lapackは)

* これらの設定を完了するために、[MinGW　32bits](http://www.mingw.org/)をダウンロードしていただく必要があります。お使いのコンピューターが64-bitであっても、こちらをダウンロードしてください。続けて、[Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)をダウンロードしてください。 

* Lapackは[VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)も必要になります。 必要であれば、 [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/)文章もご参照ください。

* *DL4Jデヴェロッパーの方:*  [Github](https://windows.github.com/)をインストールしてください。. ファイルを複製するためには以下の文章をterminal (Mac) もしくは Git Shell (Windows)へ入力してください。

      git clone https://github.com/SkymindIO/nd4j
      git clone https://github.com/SkymindIO/deeplearning4j

##<a name="source">ソースの取得方法</a>

DL4Jのソースを取得するためには [Github repo](https://github.com/SkymindIO/deeplearning4j/)にアクセスをしてください。DL4Jをよりご活用いただきたい方は、Githubをインストールしてください。[Macの方はこちら](https://mac.github.com/) or [Windowsの方はこちら](https://windows.github.com/). そしてgit cloneを行い、以下のコードをMavenに入力してください。

      mvn clean install -DskipTests -Dmaven.javadoc.skip=true

##<a name="eclipse">Eclipse</a> 

 git cloneを行った後、以下のコマンドを入力してください。

      mvn eclipse:eclipse 
  
このコマンドを入力することで、ソースをインポートしすべてをセットアップすることができます。

## <a name="trouble">トラブルシューティング</a>

*もしDL4Jを活用いただく中でトラブルが発生した場合は, [ND4J](http://nd4j.org/ja-getstarted.html)でgit　cloneを行ってください。次にND4Jでclean Maven installを行ってください。そして DL4Jを再度インストールしてください。最後にDL4Jでclean Maven installを行ってください。

      mvn clean install -DskipTests -Dmaven.javadoc.skip=true

* サンプルを実行している時に、分類作業の精確さを示す[f1 score](../glossary.html#f1)という数値が低く表示されるかと思います。 このケースでは、低い数値が低い精確性を示している訳ではありません。なぜならば、動作確認作業をの効率を高める為に、少ない量のデータセットしか与えていないからです。少ない量のデータセットは、大きな量のデータセットに比べて精確性は劣ります。例として、非常に少ない量のデータセットから生まれる精確性は、0.32から1.0の間になります。

* プログラミング内容のコピーをご覧になりたい方は、コチラ[Deeplearning4j's　classes　and　methods](http://deeplearning4j.org/doc/).をクリックしてください。

## <a name="next">Next Steps: MNISTとサンプルの実行方法</a>

[MNISTについて](../mnist-tutorial.html)をご覧ください。もうすでにdeep　learningの具体的な活用法が決定している場合には、[custom datasets](../customdatasets.html).をクリックしてください。
