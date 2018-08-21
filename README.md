# Monorepo of Deeplearning4j

Welcome to the new monorepo of Deeplearning4j that contains the source code for all the following projects, in addition to the original repository of Deeplearning4j moved to [deeplearning4j](deeplearning4j):

 * https://github.com/deeplearning4j/libnd4j
 * https://github.com/deeplearning4j/nd4j
 * https://github.com/deeplearning4j/datavec
 * https://github.com/deeplearning4j/arbiter
 * https://github.com/deeplearning4j/nd4s
 * https://github.com/deeplearning4j/gym-java-client
 * https://github.com/deeplearning4j/rl4j
 * https://github.com/deeplearning4j/scalnet
 * https://github.com/deeplearning4j/jumpy

To build everything, we can use commands like
```
./change-cuda-versions.sh x.x
./change-scala-versions.sh 2.xx
./change-spark-versions.sh x
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn clean install -Ptestresources -Ptest-native -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```

# Want some examples?
We have separate repository with various examples available: https://github.com/deeplearning4j/dl4j-examples

In the monorepo, you'll find a tutorial series in Zeppelin:

 * https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/dl4j-examples/tutorials
