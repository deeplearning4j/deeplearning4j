# Build on Mac OSX 10.11.x

Make sure you follow the instructions in order (its proved to work so far, and if this doesnt help, just open a new case, its something we are all learning):

1. Make sure you have all the Apple updates installed (at time of writing 10.11.4

2. Make sure you have the latest XCode (go to the app store and make sure you have the latest, at time of writing 7.3)
  You will also want to run `xcode-select --install` to ensure the correct version of Xcode is available.

3. Make sure you have brew (if not get it http://brew.sh)

4. run `brew install gcc5 cmake` in terminal (this will install GCC (at time of writing 5.4.0). Also please note: openblas is optional, Apple Accelerate library is supported as well.

4.1 Add following symbolic links:
```
ln -s /usr/local/Cellar/gcc5/5.4.0/bin/gcc-5 /usr/local/bin/gcc-5
ln -s /usr/local/Cellar/gcc5/5.4.0/bin/g++-5 /usr/local/bin/g++-5
ln -s /usr/local/Cellar/gcc5/5.4.0/bin/gfortran-5 /usr/local/bin/gfortran-5
```

4.2 Optional: Remove gcc-6 by running 
```
brew remove gcc
```
CMake is unable to operate if you have both gcc (currently at version 6) and gcc-5
You can restore gcc after successful build by running:
```
brew insall gcc
```

5. Clone the libnd4j repo from the deeplearning4j (and unzip if its not automatically done)

6. export the path to the folder as LIBND4J_HOME (eg: export LIBND4J_HOME=/workspace/libnd4j-master) 
6b. run `export` in terminal and make sure your path variable contains the path to `gcc-5` (`/usr/local/Cellar/gcc5/5.4.0/bin/`)
6c. make sure there are no exports for variables like LIBRARY_PATH, or LD_LIBRARY_PATH or DYLD_LIBRARY_PATH, but if they are, and are pointing to anything inside `gcc-5` remove the `gcc-5` bits (or if you have trouble, remove the paths entirely)

7. `cd $LIBND4J_HOME` 
8. `./buildnativeoperations.sh` (This will give a LOT of warnings, dont fret, if the last line is something like `[100%] Built target nd4j`)

9. make sure you have maven installed and accessible via terminal at time its available to intelliJ but but via terminal ( try `mvn --version`) if you dont have `brew install maven`

10. Now download the latest JavaCPP (https://github.com/bytedeco/javacpp) and build it using `mvn clean install -X -DskipTests -Dmaven.javadoc.skip=true`

11. Now download the latest [nd4j](https://github.com/deeplearning4j/nd4j.git) project from the repo. Make sure that nd4j is at the same leve in the tree as libnd4j, otherwise the maven will complain about not being able to find the header files.

12. inside the nd4j directory run `mvn clean install -X -DskipTests -Dmaven.javadoc.skip=true -pl '!org.nd4j:nd4j-cuda-9.0,!org.nd4j:nd4j-tests,!org.nd4j:nd4j-cuda-9.0-platform'` this should mainly fly by.

Once you have done all the above steps and have been successful, you should now open the IntelliJ ( or any other IDE) and properly set the maven dependencies for yoru project

### CPU Backend

Exchange nd4j-x86 for nd4j-native like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native</artifactId>
        <version>0.4-rc3.10-SNAPSHOT</version>
    </dependency>

and then reimport and try out.

If you follow the steps EXACTLY as written, you should be safe 
