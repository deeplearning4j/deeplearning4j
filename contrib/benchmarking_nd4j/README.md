# benchmarking_nd4j
Simple Microbenchmarks for ND4J utilizing the JMH

Original credit to https://github.com/treo/benchmarking_nd4j

## Building

   mvn clean package

## Running

  java -jar target/benchmarks.jar


For a quick run, e.g. just the same Matrix Size progression as used in the Neanderthal benchmarks, run like this:

   java -jar target/benchmarks.jar -f2 -i10 -wi 2 Neanderthal

## Running the memory profiler to detect memory leaks
Ensure jemalloc is installed with the profiler enabled. Something like:
wget https://github.com/jemalloc/jemalloc/releases/download/5.2.0/jemalloc-5.2.0.tar.bz2 && \
tar -xvf jemalloc-5.2.0.tar.bz2 && \
cd jemalloc-5.2.0 && \
./configure --enable-prof && \
make && \
make install

Run memory-prof.sh

This will run the above build process and the [MemoryPressureTest](src/main/java/org/nd4j/benchmark/memory/MemoryPressureTest.java) which will allocate a large amount of memory and then free it. The jemalloc profiler will be used to detect memory leaks.

## Choosing a BLAS Library

Since ND4J supports multiple blas libraries, you have to specify which one you actually want to use. For my own benchmarks I've been using MKL. 

    -Dorg.bytedeco.javacpp.openblas.load=mkl_rt
    
For more information see https://github.com/bytedeco/javacpp-presets/tree/master/openblas
