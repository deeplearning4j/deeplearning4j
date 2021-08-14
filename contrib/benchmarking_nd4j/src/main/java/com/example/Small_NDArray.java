package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class Small_NDArray {


    @State(Scope.Thread)
    public static class SetupState {
        public INDArray array1 = Nd4j.ones(1<<8);
        public INDArray array2 = Nd4j.ones(1<<8);

        @Setup
        public void setup(){
            NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(16384);
            NativeOpsHolder.getInstance().getDeviceNativeOps().setTADThreshold(64);
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void sumNumber(SetupState state) {
        state.array1.sumNumber().doubleValue();
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void add(SetupState state) {
        state.array1.add(state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void addi(SetupState state) {
        state.array1.addi(state.array2);
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void sub(SetupState state) {
        state.array1.sub(state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void subi(SetupState state) {
        state.array1.subi(state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void mul(SetupState state) {
        state.array1.mul(state.array2);
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void muli(SetupState state) {
        state.array1.muli(state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void cumsum(SetupState state) {
        state.array1.cumsum(0);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void cumsumi(SetupState state) {
        state.array1.cumsumi(0);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void assign(SetupState state) {
        state.array1.assign(state.array2);
    }

}
