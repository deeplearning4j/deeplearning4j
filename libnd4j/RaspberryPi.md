
### Cross compiling for rapsberry pi and android on linux

`bash pi_build.sh` using this helper script one can cross build libnd4j and dl4j with **arm COMPUTE LIBRARY** . it will download cross compiler and arm compute library.


|options | value  | description
|--|--|--|
| -a or --arch | arm32  | cross compiles for pi/linux 32bit
| -a or --arch | arm64  | cross compiles for pi/linux 64bit
| -a or --arch | android-arm  | cross compiles for android 32bit
| -a or --arch | android-arm64  | cross compiles for android 64bit
|-m or --mvn |  | if provided will build dl4j using maven

example:  
`bash pi_build.sh --arch android-arm64 --mvn`

to change version of the  **arm COMPUTE LIBRARY**  modify this line in the script
    ```
    ARMCOMPUTE_TAG=v20.05
    ```


##### old one

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
    $ git clone https://github.com/eclipse/deeplearning4j.git
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
