---
title: 本地构建DL4J堆栈
layout: cn-default
---

# 用源码进行本地构建

如果开发者和工程师想使用Deeplearning4j的最新版本，或者希望用派生项目构建自己的版本，那么请参考本页的指南来构建和安装Deeplearning4j。首选的安装目标是本地计算机上的Maven代码库。若不使用主支，可按具体需求修改下列步骤（即更换GIT分支并修改`build-dl4j-stack.sh`脚本）。

本地构建需要整个Deeplearning4j堆栈，包括：

- [libnd4j](https://github.com/deeplearning4j/libnd4j)
- [nd4j](https://github.com/deeplearning4j/nd4j)
- [datavec](https://github.com/deeplearning4j/datavec)
- [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)

Deeplearning4j能在大多数平台（Windows、OS X、Linux）上运行，同时也有多种不同的“风格”，可以适应包括CPU（OpenBLAS, MKL, ATLAS）和GPU（CUDA）在内的各种计算架构。DL4J堆栈也支持x86和PowerPC架构。

## 系统要求

在构建和安装DL4J堆栈*之前*，请确保本地计算机已具备必要的软件，且环境变量已设置完毕。各平台和操作系统版本对应的步骤可能有所不同。所需软件包括：

- git
- cmake（3.2或更高版本）
- OpenMP
- gcc（4.9或更高版本）
- maven（3.3或更高版本）

不同架构对应的软件包括：

CPU架构：

- 英特尔MKL
- OpenBLAS
- ATLAS

GPU架构：

- CUDA
- JOCL（即将支持）

### 安装必备工具

#### Linux

##### Ubuntu

若您使用Ubuntu风格的Linux，而且是以非root用户的身份运行系统，那么请按以下步骤安装必备软件：

```
sudo apt-get purge maven maven2 maven3
sudo add-apt-repository ppa:natecarlson/maven3
sudo apt-get update
sudo apt-get install maven build-essentials cmake libgomp1
```
<br />

#### OS X

可以用Homebrew来安装必备软件。如果本地计算机上已安装Homebrew，请按以下步骤安装必要的工具。

在使用Homebrew之前，首先需要确保已安装最新版本的Xcode（用作主编译器）：

```
xcode-select --install
```

最后安装必备工具：

```
brew update
brew install maven clang-omp
```
<br />

#### Windows

libnd4j的编译依赖一些Unix实用工具，因此需要安装[Msys2](https://msys2.github.io/)才能对其进行编译。

按照Msys2的[说明](https://msys2.github.io/)完成安装后，还需要安装一些附加的开发包。启动msys2 shell，用以下命令安装开发环境：

    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas

这将安装需要在msys2 shell中使用的依赖项目。

同时还需要设置PATH环境变量，加入`C:\msys64\mingw64\bin`（或自定义的msys2安装路径）。如果打开了IntelliJ或其他IDE，则必须将其重启，上述变化方会对经由IDE启动的应用程序生效。若不重启IDE，可能会出现“找不到依赖库”的错误。

### 安装必备架构

必备工具安装完毕后，即可为您的平台安装必备架构。
<br />

#### 英特尔MKL

在目前所有可用于CPU的架构中，英特尔MKL的速度最快，但在安装之前需要先完成一些“预备”工作。

1.在[英特尔的网站](https://software.intel.com/en-us/intel-mkl)上申请许可证
2.完成英特尔要求的几个步骤后，将会收到下载链接
3.下载后，按照[安装指南](https://software.intel.com/sites/default/files/managed/94/bf/Install_Guide_0.pdf)安装英特尔MKL
<br />

#### **OpenBLAS**

#### Linux

##### Ubuntu

Ubuntu用户可以用如下方式安装OpenBLAS：

```
sudo apt-get install libopenblas-dev
```

同时还需要确保`PATH`已设置为`/opt/OpenBLAS/lib`（或其他自定义的OpenBLAS主目录）。为了能在Apache Spark中使用OpenBLAS，还需要输入以下命令：

```
sudo cp libopenblas.so liblapack.so.3
sudo cp libopenblas.so libblas.so.3
```
<br />

##### CentOS

以root用户的身份在终端（或ssh会话）中输入下列命令：

    yum groupinstall 'Development Tools'

随后应当能看到终端进行许多安装活动。若要了解安装状态，比如需要确认*gcc*是否安装成功，可输入如下命令：

    gcc --version

更完整的说明请[参见此页](http://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/)。
<br />

##### OS X

您可以用Homebrew Science在OS X上安装OpenBLAS：

```
brew install homebrew/science/openblas
```
<br />

##### Windows

OpenBLAS有适用于`msys2`的安装包，可以用`pacman`命令进行安装。
<br />

#### **ATLAS**

#### Linux

##### Ubuntu

用apt安装包在Ubuntu上安装ATLAS：

```
sudo apt-get install libatlas-base-dev libatlas-dev
```
<br />

##### CentOS

在CentOS上安装ATLAS的方法：

```
sudo yum install atlas-devel
```
<br />

##### OS X

在OS X上安装ATLAS的过程较为复杂，需要的时间比较长，但大多数计算机应该都能用以下的命令完成安装。

```
wget --content-disposition https://sourceforge.net/projects/math-atlas/files/latest/download?source=files
tar jxf atlas*.tar.bz2
mkdir atlas (为ATLAS创建目录)
mv ATLAS atlas/src-3.10.1
cd atlas/src-3.10.1
wget http://www.netlib.org/lapack/lapack-3.5.0.tgz (如果ATALS下载包中包括此文件，就无需本条命令)
mkdir intel(创建build目录)
cd intel
cpufreq-selector -g performance (本条命令需要root权限，建议加入，但并非必要)
../configure --prefix=/ATLAS的安装目录路径/ --shared --with-netlib-lapack-tarfile=../lapack-3.5.0.tgz
make
make check
make ptcheck
make time
make install
```
<br />

#### CUDA

##### Linux与OS X

安装CUDA等GPU架构的详细指南请[参见此页](http://nd4j.org/gpu_native_backends.html)。
<br />

##### Windows

构建CUDA后端需要满足一些额外的条件：

* [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
* [Visual Studio 2012或2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx)（请注意：CUDA 7.5或以下版本*不支持*Visual Studio 2015）

在构建CUDA后端之前，必须调用`vcvars64.bat`来设置一些环境变量。
但首先需要将系统环境变量`SET_FULL_PATH`设为`true`，让`vcvars64.bat`设置的所有变量能够传递到mingw shell。

1.在常规命令行窗口cmd.exe中运行`C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat`
2.在其中运行`c:\msys64\mingw64_shell.bat`
3.切换至libnd4j文件夹
4.`./buildnativeoperations.sh -c cuda`

此命令将会构建CUDA nd4j.dll。


## 安装DL4J堆栈

### **OS X与Linux**

### 检查环境变量

在运行构建DL4J堆栈的代码之前，必须确保一些特定的环境变量已设置妥当。不同架构下的具体操作方法如下。
<br />

#### LIBND4J_HOME

您需要知道运行DL4J构建代码的准确路径（建议使用空目录），否则构建将会失败。确定路径后，在其末尾添加`/libnd4j`，导出到本地环境。示例如下：

```
export LIBND4J_HOME=/home/user/directory/libnd4j
```
<br />

#### 使用MKL的CPU架构

与MKL的链接可以在构建时进行，也可以在运行时同最初链接到OpenBLAS等其他BLAS实现的二进制文件进行链接。若要用MKL进行构建，只需对Linux的`LD_LIBRARY_PATH`环境变量（或者Windows的`PATH`）添加包含`libmkl_rt.so`的路径（或者Windows中包含`mkl_rt.dll`的路径），例如`/path/to/intel64/lib/`，随后按同样的方法构建。在Linux系统中，为确保使用的OpenMP版本正确，可能还需要设置下列环境变量：

```bash
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/lib64/libgomp.so.1
```

如果无法重新构建libnd4j，也可以把运行时的加载对象从OpenBLAS改为MKL库，但这种办法更复杂一些。请按下列附加步骤操作。

1.确保`/lib64/libopenblas.so.0`和`/lib64/libblas.so.3`等文件不可用（或者在Windows的`PATH`中处于靠后位置），否则它们一开始就会被libnd4j用绝对路径加载。

2.在`/path/to/intel64/lib/`中，为`libmkl_rt.so`（Windows系统为`mkl_rt.dll`）创建一个符号链接或副本，用libnd4j计划加载的文件名命名，例如：

```bash
ln -s libmkl_rt.so libopenblas.so.0
ln -s libmkl_rt.so libblas.so.3
```

```cmd
copy mkl_rt.dll libopenblas.dll
copy mkl_rt.dll libblas3.dll
```

3.最后，将`/path/to/intel64/lib/`添加至`LD_LIBRARY_PATH`环境变量（或Windows的`PATH`变量的靠前位置），然后照常运行Java应用程序。

<br />

### 构建脚本

Github社区中有一套用bash编写的[build-dl4j-stack.sh](https://gist.github.com/crockpotveggies/9948a365c2d45adcf96642db336e7df1)脚本，可以克隆DL4J stack堆栈，构建所有代码库并将其安装到本地的Maven中。这一脚本在Linux和OS X平台上都可以运行。

请仔细阅读下列说明。

CPU架构请使用以下构建脚本：

```
./build-dl4j-stack.sh
```

如果采用的是GPU后端，请改用：

```
./build-dl4j-stack.sh -c cuda
```

Scala用户可以传入二进制版本号，确保与Spark相兼容：

```
./build-dl4j-stack.sh -c cuda --scalav 2.11
```

上述构建脚本会将所有的选项和标志传递给libnd4j的`./buildnativeoperations.sh`脚本。用于这些脚本的所有标志都可以通过`build-dl4j-stack.sh`进行传递。

<br />

### 手动构建

您也可以选择手动构建DL4J堆栈的各个组件。每个软件的基本步骤包括：

1.Git克隆
2.构建
3.安装

整个流程所需的命令如下文所示，唯一的例外是，libnd4j的`./buildnativeoperations.sh`所接受的参数取决于您选用的后端类型。以下步骤必须按给出的顺序依次运行，否则会出现错误。适用于GPU的命令附在注释内，为GPU后端进行构建时，应当将适用于CPU的命令替换为注释中的命令。

``` shell
# 清除现有的代码库，确保干净的构建环境
rm -rf libnd4j
rm -rf nd4j
rm -rf datavec
rm -rf deeplearning4j

# 编译libnd4j
git clone https://github.com/deeplearning4j/libnd4j.git
cd libnd4j
./buildnativeoperations.sh
# 如果使用GPU
# ./buildnativeoperations.sh -c cuda
export LIBND4J_HOME=`pwd`
cd ..

# 构建nd4j并安装至本地的maven
git clone https://github.com/deeplearning4j/nd4j.git
cd nd4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-7.5,!:nd4j-cuda-7.5-platform,!:nd4j-tests'
## 适用于较新的0.6.1版本的上述命令
mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests'

# 如果使用GPU
# mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-tests'
cd ..

# 构建并安装datavec
checkexit git clone https://github.com/deeplearning4j/datavec.git
cd datavec
if [ "$SCALAV" == "" ]; then
  checkexit bash buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
else
  checkexit mvn clean install -DskipTests -Dmaven.javadoc.skip=true -Dscala.binary.version=$SCALAV -Dscala.version=$SCALA
fi
cd ..

# 构建并安装deeplearning4j
git clone https://github.com/deeplearning4j/deeplearning4j.git
cd deeplearning4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true
# 或者为多个Scala版本进行交叉编译
# ./buildmultiplescalaversions.sh clean install -DskipTests -Dmaven.javadoc.skip=true
## 如果跳过了CUDA，可能需要将
## -pl '!:deeplearning4j-cuda-8.0'
## 添加至mvn clean install命令，避免构建脚本寻找cuda库
cd ..
```

## 使用本地依赖项目

DL4J堆栈安装到本地的Maven代码库之后，就可以将其加入构建工具的依赖项目了。请参阅Deeplearning4j的[完全安装指南](http://deeplearning4j.org/cn/gettingstarted)，以正确的方式将版本替换为目前[主POM](https://github.com/deeplearning4j/deeplearning4j/blob/master/pom.xml)上的SNAPSHOT版本。

需要注意的是，某些构建工具，比如Gradle和SBT，无法正确调用特定平台的二进制文件。可以按[此处](http://nd4j.org/dependencies.html)的指南来设置您选用的构建工具。

## 技术支持与协助

如果您在本地构建过程中遇到任何问题，Deeplearning4j的[早期用户交流群](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters)专门针对构建问题和其他根源问题提供协助。请在Gitter上寻求帮助。
