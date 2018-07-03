# Monorepo of Deeplearning4j

Welcome to the new monorepo of Deeplearning4j that contains the source code for all the following projects, in addition to the original repository of Deeplearning4j:

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
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn clean install -Ptestresources -Ptest-native -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
Also included in the monorepo is a tutorial series in Zeppelin available in:
 * https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j/dl4j-examples/tutorials
