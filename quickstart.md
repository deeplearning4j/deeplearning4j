---
title:
layout: default
---


Use from Maven Central (Must have Maven installed)
=========================================

Include an nd4j backend here:
 <dependency>
   <groupId>org.nd4j</groupId>
   <artifactId>nd4j-$BACKEND_OF_YOUR_CHOICE</artifactId>
   <version>${nd4j.version}</version>
 </dependency>
 
THe possible backends right now are:

    nd4j-jcublas-${YOUR_CUDA_VERSION}
    nd4j-jblas
    nd4j-netlib-blas

where version can be found from [Maven Central](http://search.maven.org)

For core algorithms, you can get away with:

<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-core</artifactId>
    <version>${deeplearning4j.version}</version>
</dependency>



For nlp:

<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-nlp</artifactId>
    <version>${deeplearning4j.version}</version>
</dependency>

For scaleout (hadoop/spark):

<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-core</artifactId>
    <version>${deeplearning4j.version}</version>
</dependency>




Install from Source (YOU DO NOT HAVE TO DO THIS IF YOU ARE JUST USING THE SOFTWARE FROM MAVEN CENTRAL OR THE DOWNLOADS)
==============================
1. Download [Maven](http://maven.apache.org/download.cgi) and set it up in your path.
2. Run setup.sh on unix and setup.bat on windows



