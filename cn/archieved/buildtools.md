---
title: 配置自动化构建工具
layout: default
---

## 配置自动化构建工具

我们一般鼓励Deeplearning4j、ND4J和DataVec的用户选择Maven，但也有必要介绍如何为Ivy、Gradle和SBT等其他工具配置构建文件，这尤其是因为考虑到Google推荐使用Gradle而非Maven来构建Android项目。 

以下操作指南适用于DL4J和ND4J的所有子模块，包括deeplearning4j-api、deeplearning4j-scaleout和ND4J后端。所有项目及子模块的**最新版本**均可以在[Maven中央仓库](https://search.maven.org/)中找到。截止到2016年10月，最新版本为`0.6.0`。如用源码构建，最新版本是`0.6.1-SNAPSHOT`。

## Maven

在Maven中使用Deeplearning4j时，需要为POM.xml添加以下代码：

    <dependencies>
      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>deeplearning4j-core</artifactId>
          <version>0.6.1-SNAPSHOT</version>
          <scope>provided</scope>
      </dependency>
    </dependencies>

## Ivy

在Ivy中使用lombok时，需要为ivy.xml添加以下代码：

    <dependency org="org.deeplearning4j" name="deeplearning4j-core" rev="0.6.0" conf="build" />

## SBT

在SBT中使用Deeplearning4j时，需要为build.sbt添加以下代码：

    libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"

## Gradle

在Gradle中使用Deeplearning4j时，需要在build.gradle文件的dependencies脚本块中添加以下代码：

    provided "org.deeplearning4j:deeplearning4j-core:0.6.0"

## Leiningen

Clojure程序员可以使用与Maven相兼容的[Leiningen](https://github.com/technomancy/leiningen/)或[Boot](http://boot-clj.com/)。[Leiningen教程参见此处](https://github.com/technomancy/leiningen/blob/master/doc/TUTORIAL.md)。

注：您仍然需要下载ND4J、DataVec和Deeplearning4j，或者双击Maven / Ivy / Gradle下载的相应的JAR文件，以便在Eclipse安装过程中安装这些组件。

## 后端

[ND4J后端](http://nd4j.org/backend)及其他[依赖项](http://nd4j.org/dependencies)在ND4J网站上均有介绍。
