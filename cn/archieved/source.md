---
layout: cn-default
title: 用源代码工作
---

# 用源代码工作

如果您不打算以提交者（committer）的身份为Deeplearning4j做贡献，或者您不需要最新的Alpha版本，那么我们建议您从[Maven中央仓库](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)下载Deeplearning4j最新的稳定发布版本，0.4-rc*。具体操作与在IntelliJ中将依赖项添加至您的POM.xml一样简单。

与此同时，我们的[Github代码库见此处](https://github.com/deeplearning4j/deeplearning4j/)。请安装[Mac](https://mac.github.com/)或[Windows](https://windows.github.com/)平台的[Github](http://nd4j.org/getstarted.html)，然后将代码库“Git克隆”，让Maven运行如下命令：

      mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true

如果您希望在主线安装完成后运行Deeplearning4j示例，您应当依次先后*Git克隆* ND4J、Canova和Deeplearning4j，然后用Maven运行上述命令，从源代码安装所有的库。

按上述步骤操作，您应当就能运行0.4-rc*的示例了。 

对于已有项目而言，您可以自行构建Deeplearning4j的源文件，然后将依赖项以JAR文件的形式添加至您的项目。Deeplearning4j和[ND4J](http://nd4j.org/dependencies.html)使用的每一个依赖项都可以如此作为JAR文件添加至项目的POM.xml，您需要在`properties`标签对之间指定ND4J或Deeplearning4j的最新版本。 

        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-ui</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-nlp</artifactId>
            <version>${dl4j.version}</version>
        </dependency>

在使用源代码工作时，您需要为IntelliJ或Eclipse安装一个[Lombok项目插件](https://projectlombok.org/download.html)。

如需了解为Deeplearning4j做出贡献的方法，请阅读我们的[开发者指南](./devguide.html)。

<!-- #### <a name="one">神奇的一行命令安装法</a>

从未“Git克隆”过Deeplearning4j的用户可以用下面这一行命令来安装该学习框架及配套的ND4J和Canova：

      git clone https://github.com/deeplearning4j/deeplearning4j/; cd deeplearning4j;./setup.sh -->
