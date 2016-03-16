---
title: 完全安装Deeplearning4j
layout: zh-default
---

# 完全安装

安装过程为多步骤。如果您希望提问或反馈，我们强烈建议您加入我们的[Gitter线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)，以便我们引导您完成安装过程。即使您加入后无意发言，或更希望自行研究，也欢迎加入交流群默默潜水学习。此外，如果您对深度学习刚刚入门，我们还准备了一份[上手时需要学习的清单](../deeplearningforbeginners.html)。 

现在，请先访问我们的[快速入门页](../quickstart.html)。仅需几步即可运行我们的示例。请在完成之后，再开始本页的安装过程。如此，上手DL4J会较为容易。 

安装Deeplearning4j必备组件已列明于[ND4J入门指南页](http://nd4j.org/getstarted.html)。ND4J是驱动DL4J神经网络的代数引擎：

1.[Java 7或以上版本](http://nd4j.org/getstarted.html#java) 
2.[集成开发环境：IntelliJ](http://nd4j.org/getstarted.html#ide-for-java) 
3.[Maven](http://nd4j.org/getstarted.html#maven)

在安装这些必备组件之后，请阅读以下内容：

6.各操作系统说明
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
8.[GitHub](http://nd4j.org/getstarted.html#github)
9.<a href="#eclipse">Eclipse</a>
10.<a href="#trouble">疑难解答</a>
11.<a href="#results">可复现结果</a>
12.<a href="#next">下一步</a>

### <a name="linux">Linux</a>

* 考虑到对各类Blas库的依赖程度，必须对Blas进行原生绑定。

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (感谢@sujitpal对此所作的贡献)

欲了解更多关于OpenBlas的信息，请参见[这一部分](#open)。

* 如果GPU发生问题，则需要另外输入一行命令。首先，找到Cuda安装的目录。应与以此相似：

         /usr/local/cuda/lib64

然后，在终端中输入*ldconfig*，而后添加Cuda的文件目录。命令行应与此相似：

         ldconfig /usr/local/cuda/lib64

如果仍然无法加载Jcublas，则应在代码中添加-D参数（即JVM自变量）：

         java.library.path (settable via -Djava.librarypath=...) 
         // ^ 若需要可写目录，则将 
         -D直接添加于“<OTHER ARGS>”之后。 

如果您的IDE是IntelliJ，则应当无需这一步。 

### <a name="osx">OSX</a>

* OSX上已安装Blas。  

### <a name="windows">Windows</a>

* 虽然我们的Windows安装过程并不总是简单便利，但Deeplearning4j是少数几个对Windows社区提供实际支持的开源深度学习项目。请参见[ND4J页面Windows部分](http://nd4j.org/getstarted.html#windows)以了解更多信息。 

* 即使您的系统为64位，仍请安装[32位MinGW](http://www.mingw.org/)（下载目录见右上角），然后[使用Mingw下载预安装的动态库](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw)。 

* 安装[Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/)。（Lapack将询问您是否拥有Intel编译程序。您的回答是“没有”。）

* Lapack提供[VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke)的替代方案。您还应阅读[基本线性代数子程序（BLAS）](http://www.netlib.org/blas/)。 

* 或者，您也可以不使用MinGW，而是直接将Blas的dll文件复制到相关目录中的文件夹内。例如，MinGW的bin文件夹目录是：/usr/x86_64-w64-mingw32/sys-root/mingw/bin。欲了解Windows环境下相关目录的其他可能情况，请阅读[本StackOverflow页面上排名第一的回答](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install)。 

* 对Cygwin不提供支持。您必须从**DOS Windows**安装DL4J。  

* 运行本程序[WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100)可帮助您调试Windows安装中的问题。此处为[其输出的一个示例](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f)，供您参考。首先下载，然后打开命令行窗口／终端。使用`cd`命令回到安装目录。输入`WindowsInfo`，单击回车键。通过右键单击命令行窗口〉选择全部〉单击回车键，对输出进行复制。输出内容现已在剪贴板中。

**Windows**环境下的OpenBlas（见下）请下载[本文件](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1)。解压缩至某一目录（如`C:/BLAS`）。在您系统的`PATH`中加入上述目录。

### <a id="open">OpenBlas</a>

若要使x86后端上原生库的正常运转，需要在系统目录添加`/opt/OpenBLAS/lib`。然后在终端中输入以下命令

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

如此，可使[Spark](http://deeplearning4j.org/spark)与OpenBlas兼容工作。

如果OpenBlas运转不正常，请遵循以下步骤：

* 如果已经安装OpenBlas，首先进行卸载。
* 运行`sudo apt-get remove libopenblas-base`。
* 下载OpenBLAS的开发版。
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* 在**Linux**环境下，进一步检查`libblas.so.3`和`liblapack.so.3`的符号链接是否存在于`LD_LIBRARY_PATH`。如果不存在，则在`/usr/lib`中添加链接。符号链接（symlink）英文全名为“symbolic link”。您可以如此进行设置（其中，“-s”是将链接转为符号链接的指令）：

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"

* 其中，“from-here”是尚不存在的但正得到创建的符号链接。[本Stackoverflow页面](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux)解释了如何创建符号链接。此处为[Linux man页面](http://linux.die.net/man/1/ln)。
* 最后，重启IDE。 
* 关于如何在**Centos 6**上运行原声Blas的说明请见[本页面](https://gist.github.com/jarutis/912e2a4693accee42a94)或[本页面](https://gist.github.com/sato-cloudian/a42892c4235e82c27d0d)。

就**Ubuntu**（15.10）环境下的OpenBlas，请参见[本说明](http://pastebin.com/F0Rv2uEk)。

###<a name="eclipse">Eclipse</a> 

在运行`git clone`之后，输入以下命令：

      mvn eclipse:eclipse 
  
该命令将导入来源，并进行设置。 

根据多年使用Eclipse的经验，我们推荐具有相似界面的IntelliJ。Eclipse单一、庞大的架构常使我们及其他开发人员的代码产生奇怪错误。 

如果您使用Eclipse，应需要安装[Lombok插件](https://projectlombok.org/)。您还将需要Eclipse所用的Maven插件：[eclipse.org/m2e/](https://eclipse.org/m2e/).

Michael Depies已为[在Eclipse上安装Deeplearning4j](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/)撰写指南。

### <a name="trouble">疑难解答</a>

* 欢迎就错误信息在我们的[Gitter线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)中向我们提问。在发布问题时，请提供以下信息（如此将显著加速我们的解答速度！）：

      * 操作系统（Windows、OSX、Linux）和版本 
      * Java版本（7、8）：type java -version in your terminal/CMD
      * Maven版本：type mvn --version in your terminal/CMD
      * Stacktrace：请在Gist上发布错误代码，并将链接分享给我们：[https://gist.github.com/](https://gist.github.com/)
* 如果您曾安装过DL4J，但现在示例产生错误，则请对相关库进行升级。使用Maven进行升级时，仅需升级POM.xml文件中的版本，使之与[Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)中的最新版本相符。使用源进行升级时，可以在[ND4J](http://nd4j.org/getstarted.html)、Canova和DL4J上运行`git clone`，并以此顺序在所有三个目录内运行`mvn clean install -Dskiptests=true -Dmaven.javadoc.skip=true`。
* 在运行示例时，可能会得到较低的[f1分数](../glossary.html#f1)。这一分数评估的，是网络分类准确的可能性。在这一情况下，f1分数分数低并不表明表现不佳，因为示例是通过小数据组进行定型的。之所以数据组较小，是为了加快运行速度。因为小数据组相比大数据组较不具有代表性，所以其生成的结果也会有很大差异性。比如说，在示例数据量微小的情况下，我们的深度置信网络f1分数目前为从0.32到1.0不等。 
* Deeplearning4j包括**自动完成功能**。如果您不确定哪些命令可用，可任意按下某一字母键，将出现如下所示的下拉式菜单：
![Alt text](../img/dl4j_autocomplete.png)
* 此处为为所有用户准备的**Javadoc**：[Deeplearning4j的课程和方法](http://deeplearning4j.org/doc/)。
* 随着代码数量的增加，使用源进行安装将需要更多内存。如果在DL4J安装过程中发生`Permgen error`，则需要添加更多**堆空间**，方法是找到并更改隐藏的`.bash_profile`文件。这一文件在bash中添加环境变量。要了解具体有哪些变量，请在命令行中输入`env`。要添加更多堆空间，请在控制台输入下列命令：
      echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile
* 如3.0.4等较早的Maven版本可能会产生NoSuchMethodError等异常情况。解决办法是将Maven升级到最新版本（当前为3.3.x）。请通过输入命令行`mvn -v`检查Maven版本。
* 在安装Maven之后，您还将收到如下信息：`mvn is not recognised as an internal or external command, operable program or batch file.`（无法识别mvn为任何内部或外部命令、可执行文件或批处理文件。）此时，你需要在[Path变量](https://www.java.com/en/download/help/path.xml)中添加Maven，修改方法同对其他环境变量的修改。  
* 如果出现`Invalid JDK version in profile 'java8-and-higher':Unbounded range:[1.8, for project com.github.jai-imageio:jai-imageio-corecom.github.jai-imageio:jai-imageio-core:jar:1.3.0`错误信息，则说明Maven出现问题。请升级至3.3.x版本。
* 欲对部分ND4J依赖项进行编译，请安装C和C++所用的**开发工具**。[请参见我们的ND4J指南](http://nd4j.org/getstarted.html#devtools)
* [Java CPP](https://github.com/bytedeco/javacpp)的包含路径可能在**Windows**环境下发生问题。解决办法之一，是将Visual Studio包含目录中的标头文件放入Java运行时环境（JRE）的包含路径中（也即Java的安装路径）。如此将对standardio.h等文件产生影响。更多信息请访问[此页面](http://nd4j.org/getstarted.html#windows)。 
* 监测GPU的说明请见[此处](http://nd4j.org/getstarted.html#gpu]。
* 使用Java的重要理由之一是**[JVisualVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jvisualvm.html)**中预装的诊断工具。如果已安装Java，在命令行中输入`jvisualvm`，即可获得关于CPU、Heap、PermGen、Classes和Threads的图像信息。有用的视图：点击右上角`Sampler`标签，然后点击CPU或内存按钮以获得相关视图。 
![Alt text](../img/jvisualvm.png)
* 在使用DL4J过程中出现的部分问题可能与对机器学习概念和技术的不了解有关。我们强烈推荐所有Deeplearning4j用户使用本网站以外的资源，来了解机器学习的基本要点。在[这一页面](../deeplearningpapers.html)上，我们列出了机器和深度学习的一些学习资源。与此同时，我们也对DL4J编写了部分文档，载明了维持基本的、域特定语言的深度学习所需要的部分代码。
* 在通过**Clojure**使用`deeplearning4j-nlp`、通过Leiningen生成uberjar时，需要在`project.clj`中指定以下内容，以使`reference.conf`资源文件得到正确合并：`:uberjar-merge-with {#"\.properties$" [slurp str spit] "reference.conf" [slurp str spit]}`。请注意，.properties文件映射图中的首个条目通常为缺省项。如果未能完成上述步骤，在运行相关uberjar时将出现以下异常信息：`Exception in thread "main" com.typesafe.config.ConfigException$Missing:No configuration setting found for key 'akka.version'`。
* OSX环境下，浮动支持有很多问题。如果在运行我们的示例时，在应出现数字的地方出现NaN，则请将数据类型改为`double`。
* Java 7的分叉联接存在问题。解决方法是升级至Java 8。如果出现如下OutofMemory错误，则分叉联接是问题所在。`java.util.concurrent.ExecutionException: java.lang.OutOfMemoryError`
....`java.util.concurrent.ForkJoinTask.getThrowableException(ForkJoinTask.java:536)`

### <a name="results">可复现结果</a>

神经网络权重开始时随机生成，也就是说，模型开始时每次会在权重空间中习得一个不同位置。这可能会导致不同的局部最佳值。寻求可复现结果的用户需要使用相同的随机权重，必须在模型创建之前即进行初始化。可以通过以下命令行重新初始化相同的随机权重。

      Nd4j.getRandom().setSeed(123);

### <a name="next">后续步骤IRIS示例及安装神经网络</a>

欲开始创建神经网络，请参见[神经网络简介](http://deeplearning4j.org/neuralnet-overview.html)获得更多信息。

阅读[IRIS教程](../iris-flower-dataset-tutorial.html)以迅速上手。同时请参阅我们关于[受限玻尔兹曼机](../restrictedboltzmannmachine.html)的说明，以理解*深度置信网络*的基本机制。

根据[ND4J入门指南](http://nd4j.org/getstarted.html)上的说明创建新项目，并将必要的[POM依赖项](http://nd4j.org/dependencies.html)包括在内。
