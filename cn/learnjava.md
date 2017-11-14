---
title: 学习Java
layout: cn-default
---

# 学习Java编程

Java是全世界使用最为广泛的编程语言，也是Hadoop的语言。以下是一些可以帮助您学习Java编程方法的资源。

* [Learn Java The Hard Way（笨办法学Java）](https://learnjavathehardway.org/)
* [Java资源](http://wiht.link/java-resources)
* [Java Ranch：Java语言初学者社区](http://javaranch.com/)
* [普林斯顿Java语言编程导论](http://introcs.cs.princeton.edu/java/home/)
* [Head First Java（嗨翻Java）](http://www.amazon.com/gp/product/0596009208)
* [Java in a Nutshell（Java技术手册）](http://www.amazon.com/gp/product/1449370829)

## 系统要求

无论您要用Java开展哪种工作，我们都推荐使用下列工具。

* [Java（开发者版）](#Java) 1.7或更新版本（**仅支持64位版本**）
* [Apache Maven](#Maven)（自动构建及依赖项管理器）
* [IntelliJ IDEA](#IntelliJ)或Eclipse
* [Git](#Git)（版本控制系统）

#### <a name="Java">Java</a>

如您还没有Java 1.7或更新版本，请[在此下载Java开发工具包（JDK）](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)。若要检查已安装的Java版本是否兼容，请使用以下命令：

``` shell
java -version
```

请确保安装的是64位的Java，如果使用的是32位版本，会出现`no jnind4j in java.library.path`的错误信息。

#### <a name="Maven">Apache Maven</a>

Maven是针对Java项目的依赖管理和自动化构建工具。它与IntelliJ等IDE兼容性良好，可以让您轻松安装DL4J项目库。您可参照[官方说明](https://maven.apache.org/install.html)为您的系统[安装或更新Maven](https://maven.apache.org/download.cgi)的最新版本。若要检查已安装的Maven是否为最新版本，请使用以下命令：

``` shell
mvn --version
```

如果您使用的是Mac，可以在命令行中输入：

``` shell
brew install maven
```

Maven被Java开发者广泛使用，可以说是DL4J的必备条件。如果您此前并未从事Java开发，对Maven了解有限，请参考[Apache的Maven概述](http://maven.apache.org/what-is-maven.html)以及本站[面向非Java程序员的Maven简介](http://deeplearning4j.org/maven.html)，其中包括额外的疑难解答内容。Ivy和Gradle等[其他构建工具](../buildtools)也能够运行，但我们对Maven的支持最好。

* [Maven五分钟入门](http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html)

#### <a name="IntelliJ">IntelliJ IDEA</a>

集成开发环境（[IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)）让您能使用我们的API，只需要几个步骤就能配置神经网络。我们强烈推荐使用[IntelliJ](https://www.jetbrains.com/idea/download/)，它能与Maven相结合，有效管理依赖项。[IntelliJ社区版](https://www.jetbrains.com/idea/download/)是免费的。 

其他较为流行的IDE包括[Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html)和[Netbeans](http://wiki.netbeans.org/MavenBestPractices)。我们推荐使用IntelliJ，遇到问题时在[Gitter线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)中也更容易得到帮助。

#### <a name="Git">Git</a>

请安装[最新版本的Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)。如您已经安装了Git，您可以让Git自行升级至最新版本：

``` shell
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

想要了解基于Java的深度学习？那不妨从这里开始：

* [深度神经网络简介](./neuralnet-overview)
