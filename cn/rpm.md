---
title: Deeplearning4J与RPM
layout: cn-default
---

# Deeplearning4J与RPM

以下是用红帽软件包管理器（RPM）安装Deeplearning4J的步骤：

* 将[Spark Shell设置为一项环境变量](http://apache-spark-user-list.1001560.n3.nabble.com/Adding-external-jar-to-spark-shell-classpath-using-ADD-JARS-td1207.html)。
* 将.repo文件加入这个目录内
        {root}/etc/yum.repos.d
* 以下是repo文件中应当包含的内容：
        [dl4j.repo]
        
        name=dl4j-repo
        baseurl=http://repo.deeplearning4j.org/repo
        enabled=1

* 然后输入ND4J、Canova（向量化库）和DL4J的发布版本。例如您可以安装ND4J-jblas：

        sudo yum install nd4j-{backend}
        sudo yum install Canova-Distro
        sudo yum install Dl4j-Distro
        
JAR文件会被保存在/usr/local/Skymind目录下

* 所需的库分别位于用其项目名称（dl4j、nd4j或canova）命名的文件夹中 

        /usr/local/Skymind/dl4j/jcublas/lib
        /usr/local/Skymind/nd4j/jblas/lib
        
* 将每个库的文件夹中的JAR文件添加至Spark shell的classpath（见上文）。您可以在shell脚本中进行设置。 
* 选择一个项目，开始运行。示例项目参见[此处](https://github.com/deeplearning4j/scala-spark-examples)。
