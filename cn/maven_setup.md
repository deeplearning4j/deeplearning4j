---
title: IntelliJ中的Maven配置
layout: cn-default
---
# 使用DeepLearning4J所需的Maven配置

本页将介绍在使用DeepLearning4J之前如何设置Maven以及pom.xml文件。 

Maven依赖一个名为pom.xml的配置文件。

本页将协助您配置该文件中的依赖项。 


## 从示例开始

开发Deeplearning4J的同时，我们也在主动维护一系列实用Java示例。我们建议大家一开始先按照[快速入门指南](http://deeplearning4j.org/cn/quickstart)来下载示例，再将其作为一个IntelliJ项目安装。下载示例并验证工作环境之后，您可以从头开始新建一个项目，本页将详细介绍在使用Deeplearning4J之前如何对IntelliJ进行设置。 

## Maven中央仓库

您可以前往Maven中央仓库查看可用版本。请用以下方式搜索每个示例 

- https://mvnrepository.com/search?q=deeplearning4j-nlp
- 以此类推 

## 关键依赖项
- DL4J工具：用于配置、定型、实现各类神经网络
- ND4J库：用于操作数值数列
- DataVec库：用于数据摄取

## GPU和CPU相关的考量

矩阵计算需要消耗大量资源。GPU的矩阵处理效率要高得多。您可以配置ND4J，让神经网络高效利用GPU或CPU运行，方法是配置<nd4j.backend>nd4j-native-platform</nd4j.backend> 


## 配置pom.xml

### 首先设定一些属性。 

Maven属性是值的占位符，就像Ant中的属性一样。在POM文件内部任何位置均可以用${X}的形式引用属性值（X为属性名称）。

先列出所有属性，后面再分别引用，这样可以让pom.xml更加易读，因为所有的设置都集中到一处，一个接一个排列

``` xml
<properties>
      <nd4j.backend>nd4j-native-platform</nd4j.backend>
      <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
      <shadedClassifier>bin</shadedClassifier>
      <java.version>1.7</java.version>
      <nd4j.version>0.6.0</nd4j.version>
      <dl4j.version>0.6.0</dl4j.version>
      <datavec.version>0.6.0</datavec.version>
      <arbiter.version>0.6.0</arbiter.version>
      <guava.version>19.0</guava.version>
      <logback.version>1.1.7</logback.version>
      <jfreechart.version>1.0.13</jfreechart.version>
      <maven-shade-plugin.version>2.4.3</maven-shade-plugin.version>
      <exec-maven-plugin.version>1.4.0</exec-maven-plugin.version>
      <maven.minimum.version>3.3.1</maven.minimum.version>
    </properties>
```

### 依赖项管理

适用于有子pom文件的情况。

``` xml
<dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>${nd4j.version}</version>
      </dependency>
      <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-7.5-platform</artifactId>
        <version>${nd4j.version}</version>
      </dependency>
    </dependencies>
  </dependencyManagement>


```

### ND4J后端

此处的配置决定系统使用GPU还是CPU。请注意，所有的dependency标签都要嵌套在一对dependencies标签内，完整的示例参见本页结尾处。

``` xml
<!-- ND4J后端。每个DL4J项目都需要一个。一般将artifactId指定为"nd4j-native-platform"或者"nd4j-cuda-7.5-platform" -->
    <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>${nd4j.backend}</artifactId>
    </dependency>
```

### DL4J

Deeplearning4J核心组件包含构建多层神经网络的工具。这当然少不了。 

``` xml
 <!-- DL4J核心功能 -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

```

### DL4Jnlp 

自然语言处理工具包含在该依赖向内

``` xml

    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-nlp</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

```

### 可视化

在定型过程中实现可视化的用户界面以及HistogramIterationListener（柱状图迭代侦听器）可能需要与某一特定版本的guava库绑定。

``` xml
<!-- 强制指定使用UI/HistogramIterationListener时的guava版本 -->
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
      <version>${guava.version}</version>
    </dependency>
```

### 视频处理

相关示例包括视频处理演示。该依赖项提供所需的视频编解码器。

``` xml
    <!-- datavec-data-codec：仅用于在视频处理示例中加载视频数据 -->
    <dependency>
      <artifactId>datavec-data-codec</artifactId>
      <groupId>org.datavec</groupId>
      <version>${datavec.version}</version>
    </dependency>
```

### 生成图表

有些示例使用jfreechart库来生成图表。 

``` xml

<!-- 用于前馈/分类/MLP*和前馈/回归/RegressionMathFunctions示例 -->
    <dependency>
      <groupId>jfree</groupId>
      <artifactId>jfreechart</artifactId>
      <version>${jfreechart.version}</version>
    </dependency>
    
```

## Arbiter

超参数优化。 

``` xml
   <!-- Arbiter：用于超参数优化示例 -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>arbiter-deeplearning4j</artifactId>
      <version>${arbiter.version}</version>
    </dependency>
```    

### 完整的pom.xml示例

``` xml
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>YOURPROJECTNAME.com</groupId>
  <artifactId>YOURPROJECTNAME</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>jar</packaging>

  <name>YOURNAME</name>
  <url>http://maven.apache.org</url>

      <properties>
      <nd4j.backend>nd4j-native-platform</nd4j.backend>
      <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
      <shadedClassifier>bin</shadedClassifier>
      <java.version>1.7</java.version>
      <nd4j.version>0.6.0</nd4j.version>
      <dl4j.version>0.6.0</dl4j.version>
      <datavec.version>0.6.0</datavec.version>
      <arbiter.version>0.6.0</arbiter.version>
      <guava.version>19.0</guava.version>
      <logback.version>1.1.7</logback.version>
      <jfreechart.version>1.0.13</jfreechart.version>
      <maven-shade-plugin.version>2.4.3</maven-shade-plugin.version>
      <exec-maven-plugin.version>1.4.0</exec-maven-plugin.version>
      <maven.minimum.version>3.3.1</maven.minimum.version>
    </properties>

  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>${nd4j.version}</version>
      </dependency>
      <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-cuda-7.5-platform</artifactId>
        <version>${nd4j.version}</version>
      </dependency>
    </dependencies>
  </dependencyManagement>



  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>3.8.1</version>
      <scope>test</scope>
    </dependency>
    <!-- ND4J后端。每个DL4J项目都需要一个。一般将artifactId指定为"nd4j-native-platform"或者"nd4j-cuda-7.5-platform" -->
    <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>${nd4j.backend}</artifactId>
    </dependency>

    <!-- DL4J核心功能 -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-nlp</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

    <!-- deeplearning4j-ui用于HistogramIterationListener + 可视化：参见http://deeplearning4j.org/cn/visualization -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-ui</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

    <!-- 强制指定使用UI/HistogramIterationListener时的guava版本 -->
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
      <version>${guava.version}</version>
    </dependency>

    <!-- datavec-data-codec：仅用于在视频处理示例中加载视频数据 -->
    <dependency>
      <artifactId>datavec-data-codec</artifactId>
      <groupId>org.datavec</groupId>
      <version>${datavec.version}</version>
    </dependency>

    <!-- 用于前馈/分类/MLP*和前馈/回归/RegressionMathFunctions示例 -->
    <dependency>
      <groupId>jfree</groupId>
      <artifactId>jfreechart</artifactId>
      <version>${jfreechart.version}</version>
    </dependency>

    <!-- Arbiter：用于超参数优化示例 -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>arbiter-deeplearning4j</artifactId>
      <version>${arbiter.version}</version>
    </dependency>
  </dependencies>
</project>

```

## logback配置

Deeplearning4j不使用log4j日志机制，而是使用一种称为logback的系统，输出结果非常相似。logback通过一个名为logback.xml的文件控制 

如果类路径中没有logback.xml这一配置文件，就会出现大量警告消息。

我们的示例库目录（src/main/resources/logback.xml）中包含了一个logback.xml文件。 

您可以将该文件复制到资源目录中，您也可以按自己的需求来创建或修改文件。 

以下是logback.xml文件示例：

``` xml
<configuration>



    <appender name="FILE" class="ch.qos.logback.core.FileAppender">
        <file>logs/application.log</file>
        <encoder>
            <pattern>%date - [%level] - from %logger in %thread
                %n%message%n%xException%n</pattern>
        </encoder>
    </appender>

    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern> %logger{15} - %message%n%xException{5}
            </pattern>
        </encoder>
    </appender>

    <logger name="org.apache.catalina.core" level="DEBUG" />
    <logger name="org.springframework" level="DEBUG" />
    <logger name="org.deeplearning4j" level="INFO" />
    <logger name="org.canova" level="INFO" />
    <logger name="org.datavec" level="INFO" />
    <logger name="org.nd4j" level="INFO" />
    <logger name="opennlp.uima.util" level="OFF" />
    <logger name="org.apache.uima" level="OFF" />
    <logger name="org.cleartk" level="OFF" />



    <root level="ERROR">
        <appender-ref ref="STDOUT" />
        <appender-ref ref="FILE" />
    </root>

</configuration>


```
