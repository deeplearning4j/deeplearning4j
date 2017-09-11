---
title:Building the DL4J Stack Locally
layout: default
---

# Masterからローカル構築

**注意:クイックスタートガイドでも言及しておりますが、ほとんどのユーザーの方々にはMaven Centralのリリースを使用し、ソースから構築しないことをお勧めしています。**

（新しい機能を開発するためなど特別な事情がない限り、ソースから構築することは避けてください。カスタム層、カスタム活性化関数、カスタム損失関数などはすべてDL4Jを直接修正せずに追加できるのでソースから構築する必要はありません。ソースからの構築は非常に複雑で、何も有益なことがないことが多いためです。）

このページでは、最新のDeeplearning4jやフォークを使用して自分用のバージョンを構築したいディベロッパーやエンジニアの方々を対象として、Deeplearning4jの構築およびインストール手順をご紹介します。インストール先は、お使いのマシンのローカルMavenリポジトリが望ましいでしょう。Master branchを使用していない場合は、必要に応じて、これらの手順を変更してください（つまり、Git branchを切り替えて、`build-dl4j-stack.sh`スクリプトを変更します）。

ローカルで構築するには、Deeplearning4jの全スタックが必要です。これには、以下が含まれます。

- [libnd4j](https://github.com/deeplearning4j/libnd4j)
- [nd4j](https://github.com/deeplearning4j/nd4j)
- [datavec](https://github.com/deeplearning4j/datavec)
- [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)

Deeplearning4jは、ほとんどのプラットフォーム（Windows、OS X、Linux）で使用できるように設計されています。使用するコンピューティングアーキテクチャによっては複数の「フレーバー（派生品）」も含まれます。これには、CPU（OpenBLAS、MKL、ATLAS）、GPU（CUDA）などがあります。DL4Jのスタックはx86、PowerPCなどのアーキテクチャにも対応しています。

## 必要なもの

DL4Jを構築し、インストールするには、必須のソフトウェアや環境が「事前に」ご使用のローカルマシンに設定されている必要があります。また、動作させるための手順はご使用のプラットフォームやOSのバージョンによって異なる可能性があります。必要なソフトウェアには以下が含まれます。

- Git
- Cmake（3.2、またはそれ以降）
- OpenMP
- Gcc（4.9、またはそれ以降）
- Maven（3.3、またはそれ以降）

特定のアーキテクチャ用のソフトウェアには以下が含まれます。

CPUのオプション:

- Intel MKL
- OpenBLAS
- ATLAS

GPUのオプション:

- CUDA
- JOCL（近日中に利用可能）

特定の統合開発環境用の必要条件:

- IntelliJのLombokプラグイン

DL4Jの依存関係のテスト:

- dl4j-test-resources

### 必要なツールのインストール

#### Linux

**Ubuntu**
LinuxのフレーバーとしてUbuntuを使用している非ルートユーザーの方は必要なソフトウェアを以下の手順でインストールしてください。

```
sudo apt-get purge maven maven2 maven3
sudo add-apt-repository ppa:natecarlson/maven3
sudo apt-get update
sudo apt-get install maven build-essentials cmake libgomp1
```

#### OS X

Homebrewが必要なソフトウェアをインストールする方法として受け入れられています。ローカルにHomebrewをインストールする方は、次の手順を実行して必要なツールをインストールしてください。

まずは、Homebrewを使用する前にXcodeの最新版がインストールされているかを確認してください（主要なコンパイラとして使用します）。

```
xcode-select --install
```

最後に、必要なツールをインストールします。

```
brew update
brew install maven clang-omp
```

#### Windows

libnd4jは、コンパイルにいくつかのUnixユーティリティに依存しています。したがって、コンパイルするには、[Msys2](https://msys2.github.io/)をインストールする必要があります。

[Msys2のインストール手順](https://msys2.github.io/)にしたがってMsys2をセットアップした後、他にもいくつかの開発パッケージをインストールする必要があります。Msys2 shellを開始し、次のコマンドを入力して開発環境をセットアップします。

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas

これで、Msys2 shellで使用するために必要な依存関係がインストールされます。

PATH環境変数に`C:\msys64\mingw64\bin`（またはMsys2をインストールする場所）を入力する必要もあります。IntelliJ（または他の統合開発環境）が開いている場合は、IntelliJを再起動させて、この設定変更をアプリケーションに反映させてください。再起動しなければ、「依存ライブラリが見つかりません」といった内容のエラーが表示されるでしょう。

### 必要なアーキテクチャをインストール

必要なツールをインストールした後は、お使いのプラットフォームに必要なアーキテクチャをインストールします。

#### Intel MKL

CPUが使用可能なすべての既存のアーキテクチャのうち、現在のところ最速なのはIntel MKLです。しかし、実際にそれをインストールする前には「オーバーヘッド」が必要です。

1. [Intelのウェイブサイト](https://software.intel.com/en-us/intel-mkl)でライセンスを申請する。
2. Intelでいくつかの手順を実行すると、ダウンロードリンクが提供されます。
3. [セットアップ・ガイド](https://software.intel.com/sites/default/files/managed/94/bf/Install_Guide_0.pdf)を利用してIntel MKLをダウンロードし、インストールします。

#### OpenBLAS

##### Linux

**Ubuntu**
Ubuntuを使用している場合、以下を入力するとOpenBLASをインストールすることができます。

```
sudo apt-get install libopenblas-dev
```

また、`/opt/OpenBLAS/lib` （またはOpenBLASのホームディレクトリのどれでも）が`PATH`上にあることを確認する必要があります。OpenBLASがApache Sparkで使用できるようにするには、以下も入力してください。

```
sudo cp libopenblas.so liblapack.so.3
sudo cp libopenblas.so libblas.so.3
```

**CentOS**
ルートユーザーとして、ターミナル（またはsshセッション）に以下のコマンドを入力します。

    yum groupinstall 'Development Tools'

その後、ターミナルで様々な処理やインストールが実行されます。例えば、*gcc*がインストールされているかを調べるには、以下のコマンドを実行してください。

    gcc --version

完全ガイドは、[こちら](http://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/)をお読みください。

##### OS X

Homebrew ScienceでOS XにOpenBLASをインストールすることができます。

```
brew install homebrew/science/openblas
```

##### Windows

`msys2`のOpenBLASパッケージが利用可能です。コマンドの`pacman`を使用してインストールすることができます。

#### ATLAS

##### Linux

**Ubuntu**
Ubuntuでは、ATLAS用のaptパッケージが利用可能です。

```
sudo apt-get install libatlas-base-dev libatlas-dev
```

**CentOS**
下記のコマンドを入力するとCentOSにATLASをインストールすることができます。

```
sudo yum install atlas-devel
```

##### OS X

OS XにATLASをインストールするのは、幾分複雑で時間の掛かるプロセスです。しかし、次のコマンドを使用するとほとんどのマシンは動作します。

```
wget --content-disposition https://sourceforge.net/projects/math-atlas/files/latest/download?source=files
tar jxf atlas*.tar.bz2
mkdir atlas （ATLASのディレクトリを作成）
mv ATLAS atlas/src-3.10.1
cd atlas/src-3.10.1
wget http://www.netlib.org/lapack/lapack-3.5.0.tgz （Atlasのダウンロードにこのファイルが既に含まれていることもありますが、その場合はこのコマンドは必要ありません。）
mkdir intel（ビルドディレクトリを作成）
cd intel
cpufreq-selector -g performance （このコマンドにはルートへのアクセスが必要です。必須というわけではありませんが、入れておくのがお勧めです。）
../configure --prefix=/path to the directory where you want ATLAS installed/ --shared --with-netlib-lapack-tarfile=../lapack-3.5.0.tgz
make
make check
make ptcheck
make time
make install
```

#### CUDA

##### LinuxとOS X

GPUアーキテクチャー（CUDAなど）のインストール方法の詳細は、[こちら](http://nd4j.org/gpu_native_backends.html)を読みください。

##### Windows

CUDAのバックエンドを構築する前には、次のいくつかの追加要件を満たさなければなりません。

* [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
* [Visual Studio 2012または2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx) （注意: CUDA 7.5およびそれ以前のバージョンは、Visual Studio 2015を*サポートしていません*。）

CUDAのバックエンドを構築するには、`vcvars64.bat`を呼び出すことにより、最初にいくつかの環境変数を設定する必要があります。
しかし、まず最初にシステム環境変数の`SET_FULL_PATH`を`true`に設定してください。これにより`vcvars64.bat`が設定するすべての変数がMinGW Shellに渡されます。

1. 一般的なcmd.exeコマンドプロンプトに`C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat`を実行します。
2. その内部で`c:\msys64\mingw64_shell.bat`を実行します。
3. libnd4jフォルダに変更します。
4. `./buildnativeoperations.sh -c cuda`

これでCUDA nd4j.dllが構築されます。

#### 統合開発環境の要件

IntelliJなどの統合開発環境を使用してDeeplearning4jを構築している場合、特定のプラグインをインストールし、コードのハイライト表示に問題が発生しないようにしておく必要があります。Lombok用のプラグインもインストールする必要があります。

* IntelliJ Lombokのプラグイン: https://plugins.jetbrains.com/plugin/6317-lombok-plugin
* Eclipse Lombokのプラグイン: https://projectlombok.org/download.html にある手順に従ってください。

ScalNet（ScalaのAPI）やDL4JのUIなどのモジュールを使用したい場合、お使いの統合開発環境にScalaサポートがインストールされており、使用可能な状態である必要があります。

#### 依存関係のテスト

Deeplearning4jはテストに必要なすべてのリソースを含むリポジトリを別に設けています。これは中央のDL4Jリポジトリを軽量に保ち、Git履歴にBLOBの格納データ量がかさむのを防止するためです。DL4Jのスタックでテストを実行するには以下の手順に従ってください。

1. https://github.com/deeplearning4j/dl4j-test-resources を自分のローカルマシンに複製します。
2. `cd dl4j-test-resources; mvn install`

## DL4Jのスタックのインストール

## OS XとLinux

### 環境変数の確認

DL4Jスタックのビルドスクリプトを実行する前に、特定の環境変数が、ビルドの実行前に定義されていることを確認する必要があります。ご使用のアーキテクチャに応じて以下の手順に従ってください。

#### LIBND4J_HOME

DL4Jのビルドスクリプトを実行する正確なパスを入手する必要があります（何もない空のディレクトリを使用するのをお勧めします）。そうしなければ、ビルドは失敗します。このパスを決定したら、このパスの末尾に`/libnd4j`を追加して、自分のローカル環境にエクスポートします。以下はその例です。

```
export LIBND4J_HOME=/home/user/directory/libnd4j
```

# CPUのアーキテクチャーとMKL

構築する際やOpenBLASなど最初に別のBLASの実装とリンクしたバイナリとの実行中にMKLとリンクさせることができます。MKL用に構築するには、`libmkl_rt.so`（Windowsの場合は`mkl_rt.dll`）を含む`/path/to/intel64/lib/`などのパスをLinux（Windowsの場合は`PATH`）`LD_LIBRARY_PATH`環境変数に追加し、以前のように構築します。しかし、LinuxではOpenMPの正しいバージョンを使用するよう、以下の環境変数を設定する必要があることもあります。

```bash
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/lib64/libgomp.so.1
```

libnd4jを再構築できない場合は、事後にMKLライブラリを使用してOpenBLASの代わりにそれらをロードすることができますが、より複雑な方法です。また、以下の手順にも従ってください。

1. 必ずファイルの`/lib64/libopenblas.so.0`や`/lib64/libblas.so.3`などが利用可能でない（Windowsの場合は`PATH`に後で出現しない）ことを確認してください。あるいは、それらのファイルがそれらの絶対パスにより何より先にlibnd4jを使ってロードされるようにしてください。
2. `/path/to/intel64/lib/`の内部にシンボリックリンクか`libmkl_rt.so`（Windowsの場合は`mkl_rt.dll`）のコピーをlibnd4jがロードする名前に作成してください。以下はその例です。

```bash
ln -s libmkl_rt.so libopenblas.so.0
ln -s libmkl_rt.so libblas.so.3
```

```cmd
copy mkl_rt.dll libopenblas.dll
copy mkl_rt.dll libblas3.dll
```

3. 最後に、`/path/to/intel64/lib/`を`LD_LIBRARY_PATH`環境変数に（Windowsの場合は、先に`PATH`内に）追加し、通常通りにJavaアプリケーションを実行してください。


### スクリプトの構築

deeplearning4jのリポジトリからスクリプトの[build-dl4j-stack.sh](https://github.com/deeplearning4j/deeplearning4j/blob/master/build-dl4j-stack.sh)を使ってソースのlibndj4、ndj4、datavec、deeplearning4jからdeeplearning4jのスタック全体を構築することができます。DL4Jのスタックを複製し、各リポジトリを構築し、それらをローカルのMavenにインストールします。このスクリプトはLinuxとOS Xの両方のプラットフォームで使用可能です。

さて、次の説明へと進みましょう。 

CPUアーキテクチャーには、次のビルドスクリプトを使用します。

```
./build-dl4j-stack.sh
```

GPUバックエンドを使用している方は、以下を使用してください。

```
./build-dl4j-stack.sh -c cuda
```

[libndj4 README](https://github.com/deeplearning4j/libnd4j)の説明に従って`cc`フラグを使用すると、CUDAの構築を早めることができます。

Scalaユーザーの方は、Sparkと互換性を持たすためにバイナリバージョンを渡します。

```
./build-dl4j-stack.sh -c cuda --scalav 2.11
```

ビルドスクリプトは、すべてのオプションを渡し、libnd4jの`./buildnativeoperations.sh`スクリプトにフラグを付与します。これらのスクリプトに使用されたすべてのフラグは、`build-dl4j-stack.sh`を通じて渡すことができますす。

### 手動での構築

DL4Jのスタック内の各ソフトウェアを手動で構築することも可能です。手順は以下の通りです。

1. Git cloneする。
2. 構築する。
3. インストールする。

全体的な手順は、次のコマンドのようになります。ただし、libnd4jの`./buildnativeoperations.sh`は構築されたバックエンドに基づいてパラメータを承認します。提供された順に従ってこれらの手順を実行する必要があります。そうしなければ、エラーが発生することになります。GPU用の説明も付けてありますが、GPUのバックエンド用に構築する場合は、CPU用のコマンドに置き換える必要があります。 

``` shell
# 何もない状態から構築するために既存のリポジトリを削除します
rm -rf libnd4j
rm -rf nd4j
rm -rf datavec
rm -rf deeplearning4j

# libnd4jをコンパイルします
git clone https://github.com/deeplearning4j/libnd4j.git
cd libnd4j
./buildnativeoperations.sh
# そして/あるいはGPUを使用する場合
# ./buildnativeoperations.sh -c cuda -cc ここに使用する機器のアーキテクチャーを入力 
# つまり、GTX 1070を使用している場合、-cc 61を使用する場合
export LIBND4J_HOME=`pwd`
cd ..

# nd4jをローカルのMavenに構築、インストールします
git clone https://github.com/deeplearning4j/nd4j.git
cd nd4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-7.5,!:nd4j-cuda-7.5-platform,!:nd4j-tests'
## 上記コマンドのより最新バージョン0.6.1
mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'

# あるいはGPUを使用する場合
# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-tests'
cd ..

# datavecを構築し、インストールします
git clone https://github.com/deeplearning4j/datavec.git
cd datavec
if [ "$SCALAV" == "" ]; then
  bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
else
  mvn clean install -DskipTests -Dmaven.javadoc.skip=true -Dscala.binary.version=$SCALAV -Dscala.version=$SCALA
fi
cd ..

# deeplearning4jを構築し、インストールします
git clone https://github.com/deeplearning4j/deeplearning4j.git
cd deeplearning4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
# or cross-build across Scala versions
# ./buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
## CUDAを飛ばした場合は、 
## -pl '!:deeplearning4j-cuda-8.0' 
## をMavenのインストールコマンドに追加する必要があるかもしれません。これは、構築の際にCUDAのライブラリを探すことがないようにするためにです。
cd ..
```

## ローカルの依存関係を使用

ローカルのMavenにDL4Jのスタックをインストールした後は、自分の構築ツールにそれを含めることができます。Deeplearning4jの説明については、「[DL4Jをはじめましょう](http://deeplearning4j.org/gettingstarted)」に従い、[master POM](https://github.com/deeplearning4j/deeplearning4j/blob/master/pom.xml)にあるSNAPSHOTバージョンに入れ替えてください。

GradleやSBTなど一部の構築ツールは、プラットフォーム固有のバイナリを取り込むことができませんのでご注意ください。お気に入りの構築ツールをセットアップするには、[こちらの手順](http://nd4j.org/dependencies.html)に従ってください。

## サポート及びお手伝い

ローカルでの構築において問題が発生したときは、構築やその他のソース関連の問題解決をお手伝いするDeeplearning4j提供の[Early Adopters Channel（初心者向けチャンネル）](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters)を利用してください。困ったときは是非Gitterで質問してください。
