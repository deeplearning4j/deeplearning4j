---
title: Maven Configuration for IntelliJ
layout: default
---
# Rough Draft

Questions. 

Is this what we advise for folks setting up IntelliJ? 

Is this page complete ? 

Is this page bloated? 
* maven-shade ??
* Maven-exec ??
* freechart ??
* logback ??
* guava ??
* shaded classifier ??




# Maven Configuration for DeepLearning4J

This page describes how to set up maven, and your pom.xml file for use of DeepLearning4J. 

Maven relies on a configuration file pom.xml

This page will assist you in configuring the dependencies in that file. 


## Start with the Examples

As Deeplearning4J is developed we actively maintain a collection of working Java Examples. We suggest that anyone getting started 
begin by following the instructions to download and install the examples as a project in intellij [here](http://deeplearning4j.org/quickstart). 
Once you have downloaded the examples and verified a working environment you may want to set up a project from scratch, this page 
will walk you through setting up intellij for use with deeplearning4j. 

## Maven Central

You can go to Maven Central to view the available releases. Search examples for each 

- https://mvnrepository.com/search?q=deeplearning4j-nlp
- etc, etc, 

## Key Dependencies
- Dl4j tools to configure train and implement Neural Networks
- ND4J our library for manipulation of numerical arrays
- DataVec our library for data ingestion

## GPU CPU considerations

Matrix calculations are resource intensive. GPU's will handle the processing much more efficiently.
ND4J is configurable and will allow your Neural Net to make efficient use of either GPU or CPU by configuring
<nd4j.backend>nd4j-native-platform</nd4j.backend> 


## Configuring your pom.xml

### First set some properties. 

Maven properties are value placeholder, like properties in Ant. 
Their values are accessible anywhere within a POM by using the notation ${X}, where X is the property.

Using a list of properties and referring to them later allow easy reading of the pom.xml because 
all the settings are in one location one after another



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
### Dependency Management

This is useful if you have child poms. *note* Clarify this before publishing

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
### ND4J backend

This is where you configure the use of either GPU or CPU. Note that all the dependency 
tags will be nested in a single dependencies tag, full example at the end.

``` xml
<!-- ND4J backend. You need one in every DL4J project. Normally define artifactId as either "nd4j-native-platform" or "nd4j-cuda-7.5-platform" -->
    <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>${nd4j.backend}</artifactId>
    </dependency>
```

### DL4J

Deeplearning4J core provides the tools to build Multi Layer Neural Nets. Of course you will need that. 

``` xml
 <!-- Core DL4J functionality -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

```

### DL4Jnlp 

Natural Language Processing tools are included with this dependency

``` xml

    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-nlp</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

```
### Visualization

Our User Interface, and HistogramIterationListener that provides visualization as training progresses
may need pinned to a specific version of the guava libraries.

``` xml
<!-- Force guava versions for using UI/HistogramIterationListener -->
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
      <version>${guava.version}</version>
    </dependency>
```

### Video Processing

The examples include a video processing demonstration. This dependency provides the needed video codec.

``` xml
    <!-- datavec-data-codec: used only in video example for loading video data -->
    <dependency>
      <artifactId>datavec-data-codec</artifactId>
      <groupId>org.datavec</groupId>
      <version>${datavec.version}</version>
    </dependency>
```

### Generating Charts

A couple of our examples generate charts using the jfreechart libraries. 

``` xml

<!-- Used in the feedforward/classification/MLP* and feedforward/regression/RegressionMathFunctions example -->
    <dependency>
      <groupId>jfree</groupId>
      <artifactId>jfreechart</artifactId>
      <version>${jfreechart.version}</version>
    </dependency>
    
```

### Arbiter

Hyperparameter optimization. 

``` xml
   <!-- Arbiter: used for hyperparameter optimization examples -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>arbiter-deeplearning4j</artifactId>
      <version>${arbiter.version}</version>
    </dependency>
```    

### A complete pom.xml

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
    <!-- ND4J backend. You need one in every DL4J project. Normally define artifactId as either "nd4j-native-platform" or "nd4j-cuda-7.5-platform" -->
    <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>${nd4j.backend}</artifactId>
    </dependency>

    <!-- Core DL4J functionality -->
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

    <!-- deeplearning4j-ui is used for HistogramIterationListener + visualization: see http://deeplearning4j.org/visualization -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-ui</artifactId>
      <version>${dl4j.version}</version>
    </dependency>

    <!-- Force guava versions for using UI/HistogramIterationListener -->
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
      <version>${guava.version}</version>
    </dependency>

    <!-- datavec-data-codec: used only in video example for loading video data -->
    <dependency>
      <artifactId>datavec-data-codec</artifactId>
      <groupId>org.datavec</groupId>
      <version>${datavec.version}</version>
    </dependency>

    <!-- Used in the feedforward/classification/MLP* and feedforward/regression/RegressionMathFunctions example -->
    <dependency>
      <groupId>jfree</groupId>
      <artifactId>jfreechart</artifactId>
      <version>${jfreechart.version}</version>
    </dependency>

    <!-- Arbiter: used for hyperparameter optimization examples -->
    <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>arbiter-deeplearning4j</artifactId>
      <version>${arbiter.version}</version>
    </dependency>
  </dependencies>
</project>

```

## logback configuration

Deeplearning4j does not use the log4j logging mechanism. 
It uses a system called logback, which gives very similar-looking output. Logback is controlled by a file 
called logback.xml 

If there is no configuration file logback.xml in the classpath then you will receive a large number of warnings.

The examples repository includes a logback.xml file in the directory (src/main/resources/logback.xml). 

Copy that file to your resources directory or create/modify to suit your needs. 

Here is an example logback.xml

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

