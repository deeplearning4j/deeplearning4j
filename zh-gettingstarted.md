---
title: 
layout: zh-default
---

# 入门

本网站正在更新中，如想要获得最新的信息，[请参考](../gettingstarted.html) 

内容

* <a href="#quickstart">快速入门</a>
* <a href="#all">安装Deeplearning4j（所有系统)</a>
* <a href="#ide">Java IDE</a>
* <a href="#maven">Maven</a>
- <a href="#linux">Linux</a>
- <a href="#osx">OSX</a>
- <a href="#windows">Windows</a>
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

## ND4J ：Java虚拟机(JVM)的Numpy

Java 编写的ND4J是提供我们矩阵操作的Java科学计算引擎。[请点击这里到ND4J的入门页面](http://nd4j.org/zh-getstarted.html)，您需要安装它来运行DL4J 。 （它本身也是一个有趣的东西...... ）

## <a name="ide">Java IDE</a>

### 什么是 Java IDE

您可以利用集成开发环境：[Integrated Development Environment（ IDE）](https://zh.wikipedia.org/wiki/%E9%9B%86%E6%88%90%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83)来编辑，调试，以及建立源代码。在这里，我们建议您安装的Java版本，GitHub和Maven都会帮您处理您的依赖值。请访问我们的依赖值网页以便学习如何能用简单、轻松的方法来更改依赖值。

### 为什么您需要Java IDE

因为您想要建立一个完善的发展环境，以便让您能全心全意编辑您的代码。

### 您是否安装了Java IDE

您只需检查您的安装的程序。

### 安装

我们推荐[IntelliJ](https://www.jetbrains.com/idea/download/) ，同时它也是免费的，我们只需要社区版（Community Edition）的就可以了。

如果您喜欢， ND4J也可以在[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html)或[Netbeans](http://wiki.netbeans.org/MavenBestPractices)使用。

现在您可以简单的通过[IntelliJ](https://zh.wikipedia.org/wiki/IntelliJ_IDEA)来导入ND4J项目（或者通过Eclipse或 Netbeans）。

## <a name="maven">Maven</a>

### 什么是Maven

[Maven](https://zh.wikipedia.org/wiki/Apache_Maven)是一个能自动构建Java项目（除其他事项外）的工具 ，它能知道并帮您下载最新版本的图书馆（ ND4J .jar文件）到您的计算机或电脑，让您随时准备引用。

### 为什么您需要Maven

Maven可以让您轻易的安装ND4J和Deeplearning4j。他也能与集成开发环境：Integrated Development Environment（ IDE ），例如 IntelliJ兼容。

（有经验的Java开发者如果不想使用Maven，[您可以到我们的下载页面中的下载.jar文件](http://deeplearning4j.org/downloads.html)。对于专家来说这可能会更快，但也更复杂，因为这牵连到依赖关系。）

### 您是否安装了Maven

如果想要查看Maven是否的安装在您的计算机或电脑上，只要输入以下的文本到命令行：

        mvn --version

### 安装

[点击这里查看如何安装Maven](https://maven.apache.org/download.cgi)。下载包含Maven的最新稳定版本的压缩文件。

![Alt text](../img/maven_downloads.png) 

在同一个网页面的下方，请跟从涉及到操作系统的指示；例如：“基于Unix的操作系统（ Linux，Solaris和Mac OS X）。”如下图：

![Alt text](../img/maven_OS_instructions.png) 

现在，用你的IDE创建一个新的项目:

![Alt text](../img/new_maven_project.png) 

下面的图片将引导您完成使用Maven创造IntelliJ新项目。首先，您必须要命名您的小组和神器。

![Alt text](../img/maven2.png) 

只需在下面的屏幕上点击“下一步”，然后在下图命名您的项目（比如 “ Deeplearning4j ”）。

![Alt text](../img/maven4.png) 

现在，在新的Deelearning4j项目（IntelliJ里），您应该到 pom.xml 文件。POM 的建成需要几秒钟。当它完成时，会入下图所宣示：

![Alt text](../img/pom_before.png) 

您需要添加两个依赖值： “ [deeplearning4j-core](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j-core)”和像“ [nd4j - jblas](http://search.maven.org/#search%7Cga%7C1%7Cnd4j-jblas) ”线性代数后端。您将通过onsearch.maven.org来寻找它们。单击此屏幕上的“最新版本” 。

![Alt text](../img/search_maven_latest_version.png) 

在那里，你要复制依赖值得信息：

![Alt text](../img/latest_version_dependency.png) 

并粘贴到您的pom.xml的“dependecies”一节，请参考如下图：

![Alt text](../img/pom_after.png) 

就是这样。一旦你粘贴正确的依赖关系到pom（您也可以选择其他，如分布式深度学习的[deeplearning4j-scaleout](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j-scaleout)，或用于GPU的[nd4j - jcublas](](http://search.maven.org/#search%7Cga%7C1%7Cnd4j-jcublas)），就大功告成了。您可以在IntelliJ下（在src/main/java/文件夹）创建新的Java文件，并使用Deeplearning4j的API开始建立神经网络。（[如果您想得到一些启发，请参考我们的例子](http://deeplearning4j.org/zh-quickstart.html)）

## [Github](https://zh.wikipedia.org/wiki/GitHub)

### 什么是Github

[Github](https://zh.wikipedia.org/wiki/GitHub)只是在快速启动时运行DL4J例子，或在帮助开发DL4J框架。您没必要安装Deeplearning4j并使用其神经网络，所以如果你不打算帮助我们开发DL4J，您可能不需要它。在这种情况下，请直接到IDE.ml。

### 为什么您需要Github

如果您只想使用ND4J库，您就不需要GitHub，Maven将会处理.jar文件。但是，如果您想对ND4J或DeepLearning4J项目作出贡献，我们十分欢迎您向我们报告当中出现错误。（我们十分感谢那些已经对这个项目作出贡献的人）。

### 您是否安装了Github

您只是检查你安装的程序。

### 安装

您只需要到GitHub上下载的[Mac](https://mac.github.com/)

下载Mac ，[Windows](https://windows.github.com/)等版本的Github，然后输入以下命令到您的终端（ Mac）或 Git Shell（ Windows）中：

        git clone https://github.com/SkymindIO/deeplearning4j
        cd deeplearning4j
        git checkout 0.0.3.3

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
