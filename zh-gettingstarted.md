---
title: 
layout: zh-default
---

# 入门

内容

* <a href="#quickstart">快速入门</a>
* <a href="#all">安装Deeplearning4j（所有系统)</a>
-- <a href="#linux">Linux</a>
-- <a href="#osx">OSX</a>
-- <a href="#windows">Windows</a>
* <a href="#source">操作来源</a>
* <a href="#eclipse">Eclipse</a>
* <a href="#trouble">故障排除</a>
* <a href="#next">下一步</a>

## <a name="quickstart">快速入门</a>

[我们快速入门j将向您展示如何运行您的第一个例子](http://deeplearning4j.org/zh-quickstart.html)。

## <a name="all">完成安装：所有的操作系统</a>

DeepLearning4J需要[Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html)或以上的版本。

### 什么是Java

[Java](https://zh.wikipedia.org/wiki/Java)是我们的首选编程语言。

## 为什么您需要Java

Java将会帮您把您的代码转换成机器代码，让您可以在服务器，计算机或电脑和移动电话上跨平台工作。

### 您是否安装了Java

测试您的Java版本（也测试您是否拥有Java） ，通过键入以下文本到命令行：

    java -version

ND4J需要Java 7 才能执行，因此，如果您有较旧的Java版本，您需要安装一个新的。

### 安装

如果您的计算机（电脑）上没有安装Java 7，请到这里下载 Java开发工具包（ [JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) ）。 Java的下载方法将因不同的操作系统而异。对于较新的苹果操作系统，您会看到在在第一行的文件提及Mac OS X（每一次的更新的，jdk-7U后的数字将增加）。您讲会看到类似以下的文本：

        Mac OS X x64 185.94 MB -  jdk-7u67-macosx-x71.dmg

## ND4J ：Java的科学计算

Java 编写的ND4J是提供我们矩阵操作的Java科学计算引擎。[请点击这里到ND4J的入门页面](http://nd4j.org/zh-getstarted.html)，您需要安装它来运行DL4J 。 （它本身也是一个有趣的东西...... ）

## Github

### 什么是Github

[Github](https://zh.wikipedia.org/wiki/GitHub)是基于互联网的的一个分布式的版本控制系统。GitHub可以托管各种git库，并提供一个web界面，但与其它像 SourceForge或Google Code这样的服务不同，GitHub的独特卖点在于从另外一个项目进行分支的简易性。（点击此处查看现有的开放源代码软件的托管设施之间的比较） 。

### 为什么您需要Github

如果您只想使用ND4J库，您就不需要GitHub，Maven将会处理.jar文件。但是，如果您想对ND4J或DeepLearning4J项目作出贡献，我们十分欢迎您向我们报告当中出现错误。（我们十分感谢那些已经对这个项目作出贡献的人）。

### 您是否安装了Github

您只是检查你安装的程序。

### 安装

您只需要到GitHub上下载的[Mac](https://mac.github.com/) ，[Windows](https://windows.github.com/)等复制ND4J文件，输入以下的指令到您的终端（ Mac）或Git Shell（ Windows）中：

        git clone https://github.com/SkymindIO/deeplearning4j
        cd deeplearning4j
        git checkout 0.0.3.3

## Maven

[Maven](https://zh.wikipedia.org/wiki/Apache_Maven)是一个能自动构建Java项目（除其他事项外）的工具 ，它能知道并帮您下载最新版本的图书馆（ ND4J .jar文件）到您的计算机或电脑，让您随时准备引用。

### 为什么您需要Maven

只要一个命令，Maven可以让您同时安装ND4J和Deeplearning4j这两个项目。此外，它具有一个集成开发环境：Integrated Development Environment（ IDE ），我们将会在接下来的指示中要求您安装Maven。如果您很清楚的知道一切如何运作，您直接可以不需要通过Maven的模式进行调整，您直接通过我们的下载页面绕过它。

### 您是否安装了Maven

如果想要查看Maven是否的安装在您的计算机或电脑上，只要输入以下的文本到命令行：

        mvn --version

### 安装

点击这里查看如何安装[Maven](https://maven.apache.org/download.cgi)。

根据适用于您的操作系统的说明，例如基于UNIX操作系统（ Linux，Solaris和Mac OS X），然后下载包含Maven的最新稳定版本的压缩文件。

如果您想要开发ND4J，只要git 复制（git clone）此软件（如上所述） ，并运行ND4J目录中的Maven命令：

        mvn clean install -DskipTests -Dmaven.javadoc.skip=true

如果想要使用Maven安装其他软件，只需要您以下几个步骤：

* 首先，到你的你的根目录下 （例如deeplearning4j或nd4j）。
* 确保每个在目录和子目录里的 pom.xml 的文件都正确配置。Git clone 会帮您完成大部分的Pom安排工作。
* 在POM文件里添加可选依赖值和其他信息。请到这参阅[自述文件](https://github.com/SkymindIO/deeplearning4j/blob/master/README.md)学习如何使用依赖值处理NLP，扩展Akka和快照。
* 选择并下载适合您的[IDE](http://nd4j.org/zh-getstarted.html#ide)： - Eclipse、IntelliJ或Netbeans，然后通过Maven导入deeplearning4j项目。

除此之外，您也可以通过我们的[下载网页](http://deeplearning4j.org/download.html)安装DL4J 。如果您从我们的网页下载，那么你必须手动导入jar文件到Eclipse 、IntelliJ或Netbeans。

## <a name="linux">Linux</a>

* 由于我们依赖Jblas（Java线性代数库）的处理器 ，原生绑定的BLA是必需的。

    Fedora/RHEL
    yum -y install blas
    
    Ubuntu
    apt-get install libblas* (credit to @sujitpal)

* 如果GPU损坏了，你需要输入一个额外的指令。首先，找出Cuda本身安装的位置。它应该看起来会是类似这个样子的：

    /usr/local/cuda/lib64

然后，在终端中输入入ldconfig，然后跟随文件的路径来链接Cuda。您的命令看起来将类似这个样子：

    ldconfig /usr/local/cuda/lib64

如果您仍然无法加载Jcublas ，你需要将参数-D添加到您的代码（这是一个JVM参数） ：

    java.library.path (settable via -Djava.librarypath=...) 
    // ^ for a writable directory, then 
    -D appended directly to "<OTHER ARGS>"

如果你使用的IntelliJ作为您的IDE，这一切应该能正常运行了。

## <a name="osx">OSX</a>

OSX已经安装了Jblas。

## <a name="windows">Windows</a>

* 在[Maven](http://maven.apache.org/download.cgi)的下载页那里的 ”Window” 部分有着详细的解释如何下载Maven和Java，如何正确的配置、[设置某些环境变量](http://www.computerhope.com/issues/ch000549.htm)。
* 安装[Anaconda](http://docs.continuum.io/anaconda/install.html#windows-install)。如果您的系统不能兼容64位安装，请到同个下载页面上去下载32位的。（ Deeplearning4j取决于Anaconda使用matplotlib图形产生器 ）。
* 安装LAPACK 。 （ LAPACK会问您是否有英特尔的编译器 ，您应该没有）。
* 要完成这一步，你必须安装[32位的MinGW](http://www.mingw.org/)，不管你是否有一个64位的计算机或电脑（下载按键在位右上角） ，然后下载[Mingw的预建动态库](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)。
* LAPACK提供一个替代的[VS Studio](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)的解决方案。您也可以看看的基础线性代数子程序库（ [Basic Linear Algebra Subprograms: BLAS](http://www.netlib.org/blas/) ）的文档。

## <a name="source">操作来源</a>

对于想要深入了解DL4J的，请到我们的[Github](https://github.com/SkymindIO/deeplearning4j/)库上查看。如果你想开发Deeplearning4j ， 只需安装适用于Mac或Windows的Github 。然后 git clone 这个知识库，并运行此Maven的命令 ：

    mvn clean install -DskipTests -Dmaven.javadoc.skip=true

## <a name="eclipse">Eclipse</a>

运行一个Git clone后，输入以下命令

    mvn eclipse:eclipse 

这将导入来源并帮您设置一切。

## <a name="trouble">故障排除</a>

* 如果您在过去已经安装了DL4J，然而现在却看到例子引发错误，请在与DL4J同一个根目录上对[ND4J](http://nd4j.org/zh-getstarted.html)运行 git clone；在ND4J内安装一个全新的Maven；重新安装DL4J ；在DL4J内安装一个全新的Maven，然后看看是否能解决问题。
* 当你运行一个例子时，您可能会得到较低的[F1](http://deeplearning4j.org/glossary.html#f1)分数 ，这通常是表示神经网络的分类是准确的。在这种情况下，一个低F1值并不是表现差，这是因为在例子里我们训练较小的数据集。我们在例子里给予较小的数据集以便可以快速运行。由于小数据集的代表性比大数据集少，所以它们所产生的结果差别也很大。就拿我们的例子来说，在极小的样本数据，我们深度信念网的F1值目前是在0.32和1.0之间徘徊。
* 请到这里查找[Javadoc](http://deeplearning4j.org/doc/)列表中的Deeplearning4j教学和方法。

## <a name="next">下一步：MNIST和运行例子</a>

请看看[MNIST](http://deeplearning4j.org/mnist-tutorial.html)教程。如果您清楚的知道深度学习是如何运行，有个明确的概念您要如何操作它，请直到访我们的[自定数据集](http://deeplearning4j.org/customdatasets.html)章节。
