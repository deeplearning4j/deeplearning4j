---
title: 
layout: zh-default
---

# 快速入门

本网站正在更新中，如想要获得最新的信息，[请参考](../quickstart.html) 

本快速入门指南假设您已经安装以下软件：

1. Java 7
2. 如IntelliJ 的集成开发环境（ IDE ）
3. [Maven](../maven.html) (Java 的自动构建工具）
4. Github (可选)

如果您需要安装上述任何一个软件，[请阅读这入门指南](http://nd4j.org/getstarted.html)。

5个简单步骤使用 DL4J 

安装上述后，如果您可以按照并实行以下五个步骤，您就可以运行DL4J：

1. "git clone" 克隆 这[例子](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples) (我们目前的版本是0.0.3.3.x.), [ND4J](https://github.com/deeplearning4j/nd4j), [Canova](https://github.com/deeplearning4j/Canova) (机器学习矢量化库), [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)。
2. 使用Maven将示例作为一个项目导入IntelliJ
3. 选择一个Blas[后端](http://nd4j.org/dependencies.html)，然后导入您的POM（应该是nd4j - jblas ）
4. 从左侧的文件树中选择示例（先从DBNSmallMnistExample.java开始 ）
5. 点击实行！（这会是一个绿色按钮）

一旦您完成了，您可以尝试其他的例子，看看结果如何。

* 首先，测试您的Java版本（也测试您是否拥有Java） ，通过键入以下文本到命令行：

		java -version

* 如果您的计算机（电脑）上没有安装Java 7，请到这里下载 [Java开发工具包](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html)（ JDK ）。Java的下载方法将因不同的操作系统而异。对于较新的苹果操作系统，您会看到在在第一行的文件提及Mac OS X（每一次的更新的，jdk -7U后的数字将增加）。您讲会看到类似以下的文本：

		Mac OS X x64 185.94 MB -  jdk-7u67-macosx-x64.dmg

* 由于我们依赖Jblas（Java线性代数库）的处理器 ，原生绑定的BLA是必需的。

		OSX
		Already Installed
		
		Fedora/RHEL
		yum -y install blas
		
		Ubuntu Linux
		apt-get install libblas* (credit to @sujitpal)
		
		Windows
		See http://icl.cs.utk.edu/lapack-for-windows/lapack/

* 接下来，git 复制（git clone）DL4J的例子：

		git clone https://github.com/deeplearning4j/dl4j-0.4-examples

接下来，您可以手动导入[Maven](https://maven.apache.org/download.cgi)项目到[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html)、 /[IntelliJ](https://www.jetbrains.com/idea/help/importing-project-from-maven-model.html)或[Netbeans](http://wiki.netbeans.org/MavenBestPractices) 平台中。

* 终于到了探索真相的时刻。您现在应该会在您的您的终端/ CMD看到神经网络已经开始被训练。迭代开始时，您会看到终端/ CMD窗口画面会往下滑（在某些情况下，该程序可能需要一分钟的时间来查找资源。）。接下来，请看看右下角第二个的行数，这个行数在每个新的迭代都会减少。这是测量当神经网络重组数字图像时的错误。神经网络正在学习时，您会看到产生的错误会逐渐减少。

![Alt text](../img/learning.png)

如果想要判断您的神经网络是否成功了解到MNIST数据集，其方法就是要看可视化。它们应该逐渐变成类似于手写的数字。当他们变成类似于手写的数字时，您的神经网络已经成功受训，这就是为什么您需要深度学习，您也能了解深度学习的强大。

至此，您应该拥有有一个能产生相对准确高的神经网络。恭喜您。 （如果您还没有得到结果，[请马上告知我们](mailto:chris@skymind.io)！ ）

一旦您已经探索了我们所有的例子，您可以按照在我们的入门页面里的指示来运行整个代码库。
