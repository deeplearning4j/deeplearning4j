---
title: "Iterative Reduce With DL4J on Hadoop and Spark"
layout: default
---

# 하둡과 스파크 기반의 DL4J를 사용한 반복적인 리듀스(Iterative Reduce) 작업

맵리듀스를 이해한다면 반복적인 리듀스 작업을 이해하는 데 더 수월할 것입니다.

## 맵리듀스(MapReduce)

맵리듀스는 큰 데이터 집합을 여러개의 코어에서 동시에 처리하기 위한 기술입니다. 구글의 Jeff Dean은 이 이론을 [2004 research paper](https://static.googleusercontent.com/media/research.google.com/en/us/archive/mapreduce-osdi04.pdf) 에서 소개했고, 이후 Doug Cutting은 Yahoo에서 이와 유사한 구조를 구현했습니다. Cutting의 프로젝트는 이후 [Apache Hadoop](https://hadoop.apache.org/) 으로 발전했습니다. 두 프로젝트는 웹 인덱싱을 처리하기 위해 사용되었고, 이후 다른 여러 어플리케이션에도 사용되고 있습니다.

맵리듀스라는 단어는 함수형 프로그래밍에서 유래된 2개의 메서드를 참조하였습니다. 맵(Map)은 목록에 있는 모든 값들에 동일한 처리(수식 계산 등)를 적용하고 적용된 새로운 값들의 목록을 생성하는 명령입니다. 리듀스(Reduce)는 여러개의 값들을 가진 목록에서 내부의 값들을 조합하여 값들의 갯수를 줄이는 명령입니다.

예를 들어, 가장 단순한 형태로 맵과 리듀스를 이해하기 위해 문장 안에 있는 각 단어의 개수를 세는 예제를 보여드리겠습니다. 주어진 문장에서, 맵은 모든 단어를 잘라내서 키-값 쌍으로 만들고 모든 단어에 1이라는 값을 반영하는 과정입니다. 리듀스 과정에서는 잘라낸 값들을 보면서 같은 단어가 발견될 경우 1로 입력한 값들을 더해가도록 하면서 각 단어가 몇개가 있는지 합계를 구합니다.

실제로 맵리듀스는 이보다 큰 스케일의 환경에서 적용되고 있습니다. 맵 과정에서는 데이터를 여러 코어로 분산시킨 뒤, 나눠진 데이터 파편들에 같은 명령을 수행시키는 방법으로 큰 단위의 작업을 분리합니다. 리듀스 과정에서는 분리되어 변형된 데이터 조각들을 통합하여 하나의 데이터 셋으로 만들고 한곳으로 모아 추가적인 작업을 수행합니다. Map explodes and Reduce collapses, like a star expands to become a Red Giant, and shrinks to a White Dwarf. 

## Iterative MapReduce

While a single pass of MapReduce performs fine for many use cases, it is insufficient for machine- and deep learning, which are iterative in nature, since a model "learns" with an optimization algorithm that leads it to a point of minimal error over many steps. 

You can think of Iterative MapReduce, also [inspired by Jeff Dean](https://static.googleusercontent.com/media/research.google.com/en/us/people/jeff/CIKM-keynote-Nov2014.pdf), as a YARN framework that makes multiple passes on the data, rather than just one. While the architecture of Iterative Reduce is different from MapReduce, on a high level you can understand it as a sequence of map-reduce operations, where the output of MapReduce1 becomes the input of MapReduce2 and so forth. 

Let's say you have a deep-belief net that you want to train on a very large dataset to create a model that accurately classifies  your inputs. A deep-belief net is composed of three functions: a scoring function that maps inputs to classifications; an error function that measures the difference between the model's guesses and the correct answer; and optimization algorithm that adjusts the parameters of your model until they make the guesses with the least amount of error. 

*Map* places all those operations on each core in your distributed system. Then it distributes batches of your very large input dataset over those many cores. On each core, a model is trained on the input it receives. *Reduce* takes all those models and averages the parameters, before sending the new, aggregate model back to each core. Iterative Reduce does that many times until learning plateaus and error ceases to shrink. 

The image, [created by Josh Patterson](http://www.slideshare.net/cloudera/strata-hadoop-world-2012-knitting-boar), below compares the two processes. On the left, you have a close-up of MapReduce; on the right, of Iterative. Each "Processor" is a deep-belief network at work, learning on batches of a larger dataset; each "Superstep" is an instance of parameter averaging, before the central model is redistributed to the rest of the cluster. 

![Alt text](./img/mapreduce_v_iterative.png)

## Hadoop & Spark

Both Hadoop and Spark are distributed run-times that perform a version of MapReduce and Iterative Reduce. Deeplearning4j works as a job within Hadoop/YARN or Spark. It can be called, run and provisioned as a YARN app, for example.

In Hadoop, Iterative Reduce workers sit on the their splits, or blocks of HDFS, and process data synchronously in parallel, before they send the their transformed parameters back to the master, where the parameters are averaged and used to update the model on each worker's core. With MapReduce, the map path goes away, but Iterative Reduce workers stay resident. This architecture is roughly similar to Spark.

To provide a little context about the state-of-the-art, both Google and Yahoo operate parameter servers that store billions of parameters which are then distributed to the cluster for processing. Google's is called the Google Brain, which was created by Andrew Ng and is now led by his student Quoc Le. Here's a rough picture of the Google production stack circa 2015 to show you how MapReduce fits in.

![Alt text](./img/google_production_stack.png)

Deeplearning4j considers distributed run-times to be interchangeable (but not necessarily equal); they are all simply a directory in a larger modular architecture that can be swapped in or out. This allows the overall project to evolve at different speeds, and separate run-times from other modules devoted to neural net algorithms on the one hand, and hardware on the other. Deeplearning4j users are also able to build a standalone distributed architecture via Akka, spinning out nodes on AWS.

Every form of scaleout including Hadoop and Spark is included in our [scaleout repository](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout).

Lines of Deeplearning4j code can be intermixed with Spark, for example, and DL4J operations will be distributed like any other. 
