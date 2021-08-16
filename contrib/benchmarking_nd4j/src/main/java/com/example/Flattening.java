package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class Flattening {

    @State(Scope.Thread)
    public static class SetupState {
        public INDArray small_c = org.nd4j.linalg.factory.Nd4j.create(new int[]{1<<10, 1<<10}, 'c');
        public INDArray small_f = org.nd4j.linalg.factory.Nd4j.create(new int[]{1<<10, 1<<10}, 'f');
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_CC_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('c', state.small_c);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_CF_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('f', state.small_c);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_FF_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('f', state.small_f);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void toFlattened_FC_Small(SetupState state) throws IOException {
        org.nd4j.linalg.factory.Nd4j.toFlattened('c', state.small_f);
    }

}
