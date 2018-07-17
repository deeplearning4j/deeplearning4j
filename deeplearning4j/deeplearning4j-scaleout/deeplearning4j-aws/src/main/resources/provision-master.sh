#!/bin/bash

################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

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


