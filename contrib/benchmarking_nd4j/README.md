# benchmarking_nd4j
Simple Microbenchmarks for ND4J utilizing the JMH

Original credit to https://github.com/treo/benchmarking_nd4j

## Building

   mvn clean package

## Running

  java -jar target/benchmarks.jar


For a quick run, e.g. just the same Matrix Size progression as used in the Neanderthal benchmarks, run like this:

   java -jar target/benchmarks.jar -f2 -i10 -wi 2 Neanderthal

## Choosing a BLAS Library

Since ND4J supports multiple blas libraries, you have to specify which one you actually want to use. For my own benchmarks I've been using MKL. 

    -Dorg.bytedeco.javacpp.openblas.load=mkl_rt
    
For more information see https://github.com/bytedeco/javacpp-presets/tree/master/openblas
