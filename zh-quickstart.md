---
title: 
layout: zh-default
---

# 快速入门

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

* 由于DL4J的数据可视化和调试采用跨平台的工具来呼叫Python程式，您也必须要拥有[Anaconda](http://continuum.io/downloads)科学计算包（点击这里下载）。安装了Anaconda科学计算包后，您可以通过Python窗口中输入以下文本以测试您是拥有否有必要的科学计算包：

		import numpy
		import pylab as pl

![Alt text](../img/python_shot.png)

当您在训练神经网络时，这些工具将产生可视化以便让您能调试神经网络。 如果您看到正常化分布，这将会是一个好兆头。这些可视化偶尔会在苹果操作系统上产生错误，但这并不会使神经网络的训练停止。

* 接下来，git 复制（git clone）DL4J的例子：

		git clone https://github.com/SkymindIO/dl4j-examples

接下来，您可以手动导入[Maven](https://maven.apache.org/download.cgi)项目到[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html)、 /[IntelliJ](https://www.jetbrains.com/idea/help/importing-project-from-maven-model.html)或[Netbeans](http://wiki.netbeans.org/MavenBestPractices) 平台中。

* 当您在IntelliJ平台打开DL4J的样本项目，去寻找MNIST的例子，然后按运行。如果MNIST演示时产生过多的渲染，使其速度减缓，这时您可以通过增加渲染呼叫的参数，保存其文件并重新启动演示。

* 终于到了探索真相的时刻。您现在应该会在您的您的终端/ CMD看到神经网络已经开始被训练。迭代开始时，您会看到终端/ CMD窗口画面会往下滑（在某些情况下，该程序可能需要一分钟的时间来查找资源。）。接下来，请看看右下角第二个的行数，这个行数在每个新的迭代都会减少。这是测量当神经网络重组数字图像时的错误。神经网络正在学习时，您会看到产生的错误会逐渐减少。

![Alt text](../img/learning.png)

* 在整个训练中，您应该看到一些数字图像小窗口在屏幕的左上角弹出。这些神经网络重组数字图像（参考下图）都是在证明您的神经网络在正常运行着。

![Alt text](../img/numeral_reconstructions.png)

如果想要判断您的神经网络是否成功了解到MNIST数据集，其方法就是要看可视化。它们应该逐渐变成类似于手写的数字。当他们变成类似于手写的数字时，您的神经网络已经成功受训，这就是为什么您需要深度学习，您也能了解深度学习的强大。

至此，您应该拥有有一个能产生相对准确高的神经网络。恭喜您。 （如果您还没有得到结果，[请马上告知我们](mailto:chris+zh@skymind.io)！ ）

一旦您已经探索了我们所有的例子，您可以按照在我们的入门页面里的指示来运行整个代码库。
