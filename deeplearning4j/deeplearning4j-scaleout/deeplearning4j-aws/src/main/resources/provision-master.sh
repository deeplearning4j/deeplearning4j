#!/bin/bash

sudo yum -y install blas java-1.7.0-openjdk-devel


if [ ! -f "/opt/dl4j" ];
then
   sudo mkdir /opt/dl4j
   sudo yum -y install git
   git clone https://github.com/agibsonccc/java-deeplearning

   wget http://www.trieuvan.com/apache/maven/maven-3/3.1.1/binaries/apache-maven-3.1.1-bin.tar.gz
   sudo mv apache-maven-3.1.1-bin.tar.gz /opt
   cd /opt && sudo tar xvf apache-maven-3.1.1-bin.tar.gz && sudo mv apache-maven-3.1.1 /opt/mvn
   cd /home/ec2-user/java-deeplearning/ && /opt/mvn/bin/mvn -DskipTests clean install
   echo "Printing distribution"
   ls /home/ec2-user/java-deeplearning/deeplearning4j-distribution/target
   echo "Before moving distribution" 
  sudo mv deeplearning4j-distribution/target/deeplearning4j-dist.tar.gz /opt
   echo "Moving distribution to opt directory..."
  echo "Moving in to opt directory"  
  cd /opt 

  sudo tar xzvf deeplearning4j-dist.tar.gz 
  #sudo mkdir /opt/dl4j 
  echo "Moving jars in to /opt/dl4j/..."
  sudo mv *.jar /opt/dl4j
fi


