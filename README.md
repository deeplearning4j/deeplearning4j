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
 * https://github.com/deeplearning4j/pydl4j
 * https://github.com/deeplearning4j/jumpy
 * https://github.com/deeplearning4j/pydatavec
 

To build everything, we can use commands like
```
./change-cuda-versions.sh x.x
./change-scala-versions.sh 2.xx
./change-spark-versions.sh x
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn -B -V -U clean install -pl '!jumpy,!pydatavec,!pydl4j' -Dlibnd4j.platform=linux-x86_64 -Dlibnd4j.chip=cuda -Dlibnd4j.cuda=9.2 -Dlibnd4j.compute=<your GPU CC> -Djavacpp.platform=linux-x86_64 -Dmaven.test.skip=true
```

An example of GPU "CC" or compute capability is 61 for Titan X Pascal.

# Want some examples?
We have separate repository with various examples available: https://github.com/deeplearning4j/dl4j-examples

In the examples repo, you'll also find a tutorial series in Zeppelin: https://github.com/deeplearning4j/dl4j-examples/tree/master/tutorials
