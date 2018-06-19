Please follow following instructions to build nd4j for raspberry PI:

1. download cross compilation tools for Raspberry PI

    ```
    $ apt-get/yum install git rsync cmake
    $ mkdir $HOME/raspberrypi
    $ cd $HOME/raspberrypi
    $ git clone git://github.com/raspberrypi/tools.git
    $ export PATH=$PATH:$HOME/raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin
    $ mkdir $HOME/raspberrypi/rootfs
    (Replace the ip address in the following command with the IP of a raspberry pi device)
    $ rsync -rl --delete-after --safe-links pi@(192.168.1.PI):/{lib,usr} $HOME/raspberrypi/rootfs
    ```

2. download deeplearning4j:

    ```
    $ cd $HOME
    $ git clone https://github.com/deeplearning4j/deeplearning4j.git
    ```

3. build libnd4j:

    ```
    $ cd deeplearning4j/libnd4j
    $ ./buildnativeoperations.sh -o linux-armhf"
    ```

4. build nd4j

    ```
    $ export LIBND4J_HOME=<pathTond4JNI>
    $ cd $HOME/deeplearning4j/nd4j
    $ mvn clean install -Djavacpp.platform=linux-armhf -Djavacpp.platform.compiler=$HOME/raspberrypi/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ -DskipTests  -Dmaven.javadoc.skip=true  -pl '!:nd4j-cuda-9.1,!:nd4j-cuda-9.1-platform,!:nd4j-tests'
    ```
