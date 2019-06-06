Please follow following instructions to build nd4j for raspberry PI:

1. download cross compilation tools for Raspberry PI

    ```
    $ apt-get/yum install git cmake
    (You may substitute any path you prefer instead of $HOME/raspberrypi in the following two steps)
    $ mkdir $HOME/raspberrypi
    $ export RPI_HOME=$HOME/raspberrypi
    $ cd $RPI_HOME
    $ git clone git://github.com/raspberrypi/tools.git
    $ export PATH=$PATH:$RPI_HOME/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin
    ```

2. download deeplearning4j:

    ```
    $ cd $HOME
    $ git clone https://github.com/deeplearning4j/deeplearning4j.git
    ```

3. build libnd4j:

    ```
    $ cd deeplearning4j/libnd4j
    $ ./buildnativeoperations.sh -o linux-armhf
    ```

4. build nd4j

    ```
    $ export LIBND4J_HOME=<pathTond4JNI>
    $ cd $HOME/deeplearning4j/nd4j
    $ mvn clean install -Djavacpp.platform=linux-armhf -Djavacpp.platform.compiler=$HOME/raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ -DskipTests  -Dmaven.javadoc.skip=true  -pl '!:nd4j-cuda-9.1,!:nd4j-cuda-9.1-platform,!:nd4j-tests'
    ```
