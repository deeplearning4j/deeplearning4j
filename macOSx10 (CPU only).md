#Build on Mac OSX 10.11.x

Make sure you follow the instructions in order (its proved to work so far, and if this doesnt help, just open a new case, its something we are all learning):

1. Make sure you have all the Apple updates installed (at time of writing 10.11.4

2. Make sure you have the latest XCode (go to the app store and make sure you have the latest, at time of writing 7.3)
  Now that that is done proceed...

3. Make sure you have brew (if not get it http://brew.sh)

4. run `brew install clang-omp` in terminal (this will install clang (at time of writing 3.5.0)

5. Clone the libnd4j repo from the deeplearning4j (and unzip if its not automatically done)

6. export the path to the folder as LIBND4J_HOME (eg: export LIBND4J_HOME=/workspace/libnd4j-master) 
6b. run `export` in terminal and make sure your path variable contains the path to `clang-omp` (`/usr/local/Cellar/clang-omp/2015-04-01/bin`)
6c. make sure there are no exports for variables like LIBRARY_PATH, or LD_LIBRARY_PATH or DYLD_LIBRARY_PATH, but if they are, and are pointing to anything inside clang-omp remove the clang-omp bits (or if you have trouble, remove the paths entirely)

7. `cd $LIBND4j_HOME` 
8. `./buildnativeoperations.sh blas cpu` (This will give a LOT of warnings, dont fret, if the last line is something like `[100%] Built target nd4j`)

9. Now download the latest nd4j files from the repo

10. make sure you have maven installed and accessible via terminal at time its available to intelliJ but but via terminal ( try `mvn --version`) if you dont have `brew install maven`

11. inside the nd4j directory run `mvn clean install -X -DskipTests -Dmaven.javadoc.skip=true -pl '!org.nd4j:nd4j-cuda-7.5'` this should mainly fly by and only fail at the end with the nd4j tests this is okay

Once you have done all the above steps and have been successful, you should now open the IntelliJ ( or any other IDE) and properly set the maven dependencies for yoru project

### CPU Backend

Exchange nd4j-x86 for nd4j-native like that:

    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native</artifactId>
        <version>0.4-rc3.9-SNAPSHOT</version>
    </dependency>

and then reimport and try out.

If you follow the steps EXACTLY as written, you should be safe 
