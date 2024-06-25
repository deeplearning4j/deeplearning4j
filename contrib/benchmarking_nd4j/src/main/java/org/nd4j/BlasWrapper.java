package org.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class BlasWrapper {

    @State(Scope.Thread)
    public static class SetupState {
        public INDArray array1 =  Nd4j.ones(100).addi(0.01f);
        public INDArray array2 =  Nd4j.ones(100).addi(0.01f);
        public INDArray array3 =  Nd4j.ones(100).addi(0.01f);


        public org.nd4j.linalg.factory.BlasWrapper wrapper = Nd4j.getBlasWrapper();

    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void asum(SetupState state) {
        state.wrapper.asum(state.array1);
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void axpy(SetupState state) {
       state.wrapper.axpy(new Float(0.75f), state.array1, state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void copy(SetupState state) {
        state.wrapper.copy(state.array1, state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void dot(SetupState state) {
        state.wrapper.dot(state.array1, state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nrm2(SetupState state) {
        state.wrapper.nrm2(state.array1);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void iamax(SetupState state) {
        state.wrapper.iamax(state.array1);
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void swap(SetupState state) {
        state.wrapper.swap(state.array1, state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void gemv(SetupState state) {
        state.wrapper.gemv((Number) new Float(0.75f), state.array1, state.array2, new Double(0.5), state.array3);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void ger(SetupState state) {
        state.wrapper.ger(new Double(0.75f), state.array1, state.array2, state.array3);
    }
}
