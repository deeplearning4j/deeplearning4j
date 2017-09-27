---
title: 用POWER处理器运行Deeplearning4j
layout: cn-default
---

用POWER处理器运行Deeplearning4j
----------------------

[POWER架构](https://en.wikipedia.org/wiki/POWER8)是一种由IBM设计的常用硬件架构。这种适用于高端服务器的处理器架构非常
适合运行深度学习。近来POWER增加了[nvlink](http://www.nvidia.com/object/nvlink.html)，
正迅速成为深度学习应用的首选CPU架构。

添加名为[nd4j-native-platform](http://repo1.maven.org/maven2/org/nd4j/nd4j-native-platform/)的[ND4J后端](http://nd4j.org/backend.html)之后，Deeplearning4j就能在POWER处理器上运行，无需更改任何代码。像其他普通的JVM项目那样声明POM.xml文件中列出的最新版本即可。

为何使用Maven（或Gradle、SBT……）
-------------------------------

本页主要提供Maven的说明，但各种自动化构建工具使用的许多术语都很相似。[此处](http://www.slideshare.net/fabiofumarola1/3-maven-gradle-and-sbt)的演示文稿将上述三种工具进行了对比。 

之所以采用自动化构建工具而非操作系统级软件包管理器，是因为Java本身是一种跨平台语言。这既有好处也有坏处。Maven及其相关工具拥有名为Maven中央仓库的专用存储库，负责处理依赖项的发布。Java IDE与这些工具的集成性非常好。基于Linux的软件包管理器通常无法很好地映射至Java依赖项，这主要是依赖项数量较多的缘故。

如果您需要构建一个在POWER服务器上运行的应用程序，我们推荐您改用uber JAR。一种简便的方法是将uber JAR作为RPM或DEB包的一部分来使用。这会将DL4J的部署与应用程序部署分离开来。

其他示例
----------------------

我们还提供在[GPU](https://deeplearning4j.org/cn/gpu)和[Android系统](https://deeplearning4j.org/cn/android)上运行DL4J的操作指南。

我们的所有[示例](https://github.com/deeplearning4j/dl4j-examples)应当都能“开箱即用”，可以直接运行。这是因为`nd4j-native-platform`捆绑了所有原生依赖项（包括POWER）。如需进一步了解有关运行示例的信息，请参见我们的[快速入门指南](http://deeplearning4j.org/cn/quickstart)。

如需在服务器上运行DL4J，您可以用Maven创建一个[uber JAR](http://stackoverflow.com/questions/11947037/what-is-an-uber-jar)，这一步骤很容易完成。

在示例中，我们用[Maven Shade插件](https://maven.apache.org/plugins/maven-shade-plugin/)来将所有必需的依赖项打包成一个JAR文件。具体操作方式可参见[此处](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/pom.xml#L140)的示例。

如果您在用POWER处理器运行Deeplearning4j时遇到任何问题，请随时在[线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)中向我们提出。
