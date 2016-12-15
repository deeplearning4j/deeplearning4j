---
title: 面向Python程序员的Maven简介
layout: default
---

# 面向Python程序员的Maven简介

[Maven](https://en.wikipedia.org/wiki/Apache_Maven)是Java程序员最常用的自动化构建工具。Python没有功能同Maven完全一致的工具，但可以认为Maven大致相当于[pip](https://en.wikipedia.org/wiki/Pip_(package_manager))这样的包管理系统，或者PyBuilder、[Distutils](http://docs.activestate.com/activepython/3.2/diveintopython3/html/packaging.html)。 

Maven也是安装运行Deeplearning4j最简便的方式，它所提供的[Scala API](http://nd4j.org/scala.html)有着会让不少Python程序员感到似曾相识的语法，同时也具备各种强大的功能。 

作为一种自动化构建工具，Maven把源代码编译为字节码，将目标文件链接生成可执行文件、库文件等。Maven的交付物是一个JAR文件，用Java源代码以及程序部署所需的资源创建。 

([JAR](https://en.wikipedia.org/wiki/JAR_%28file_format%29)文件即*Java归档文件*，英语为*Java ARchive*，是一种软件包文件格式，用于聚合大量的Java类文件、相关的元数据以及文本和图像等资源。JAR是一种压缩文件格式，帮助Java运行时部署一组类和与之相关的资源。） 

Maven会从它的中央仓库动态下载所需的Java库和Maven插件，这些库和插件由存储项目对象模型（Project Object Model）的XML文件（即POM.xml）指定。 

![Alt text](./img/maven_schema.png)

*《Maven权威指南》*中提到： 

		从命令行运行mvn install，将处理资源文件，编译源代码，运行单元测试，创建一个JAR，然后把这个JAR安装到本地仓库，以供其他项目重复使用。 

与Deeplearning4j一样，Maven遵循约定优于配置的原则，这也就是说，Maven会为项目提供一系列默认值，在运行时程序员无需为每个新项目指定所有的参数。 

如果同时安装了IntelliJ和Maven，IntelliJ允许用户在IDE中创建新项目时选用Maven，并且有向导帮助您进行设置（我们的[快速入门指南](https://deeplearning4j.org/cn/zh-quickstart)中有更为详细的介绍）。如此一来，构建工作就可以全部在IntelliJ当中完成。 

或者也可以用下列命令在项目的根目录下使用Maven，对项目执行干净构建：

		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		
上述命令让Maven在运行安装之前清除所有已编译文件目录，确保对项目进行从零开始的干净构建。


目前已有数本关于Apache Maven的书籍可供参考，可以在支持这一开源项目的Sonatype公司的网站上找到。 

### Maven疑难解答

* 3.0.4等较早的Maven版本可能会产生NoSuchMethodError等异常情况。解决办法是将Maven升级到最新版本。 
* 在安装Maven之后，您还有可能收到如下信息：*`mvn is not recognised as an internal or external command, operable program or batch file.`* （无法识别mvn为任何内部或外部命令、可执行文件或批处理文件。）这说明需要在[PATH变量](https://www.java.com/en/download/help/path.xml)中添加Maven，修改方法和其他环境变量一样。 
* 随着DL4J代码数量的增加，使用源进行安装将需要更多内存。如果在DL4J构建过程中发生Permgen错误，则需要添加更多堆空间，方法是找到并更改隐藏的`.bash_profile`文件。这一文件在bash中添加环境变量。要了解具体有哪些变量，请在命令行中输入*env*。要添加更多堆空间，请在控制台输入下列命令：
      echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile

### 扩展阅读：

* [Maven by Example（Maven实例教程）](https://books.sonatype.com/mvnex-book/reference/public-book.html)
* [Maven: The Complete Reference（Maven权威指南）](https://books.sonatype.com/mvnref-book/reference/public-book.html)
* [Developing with Eclipse and Maven（用Eclipse和Maven进行开发）](https://books.sonatype.com/m2eclipse-book/reference/)
