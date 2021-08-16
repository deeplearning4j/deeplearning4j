package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class NativeOps {

    @State(Scope.Thread)
    public static class SetupState {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);
        INDArray arrayColumn = Nd4j.linspace(1, 1024, 1024).reshape(1024,1);
        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);

        INDArray array3 = Nd4j.ones(128, 256);
        INDArray arrayRow3 = Nd4j.linspace(1, 256, 256);

        INDArray arrayUnordered = Nd4j.ones(512, 512);
        INDArray arrayOrderedC = Nd4j.zeros(512, 512,'c');
        INDArray arrayOrderedF = Nd4j.zeros(512, 512, 'f');

        {
            float sum = (float) array.sumNumber().doubleValue();
        }
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void broadcastColumn(SetupState state) throws IOException {
        state.array.addiColumnVector(state.arrayColumn);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void broadcastRow(SetupState state) throws IOException {
        state.array.addiRowVector(state.arrayRow);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void transformOp(SetupState state) throws IOException {
        Nd4j.getExecutioner().exec(new Exp(state.array1, state.array2));
    }



    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void scalarOp2(SetupState state) throws IOException {
        Nd4j.getExecutioner().exec(new ScalarMultiplication(state.arrayUnordered, 2.5f));
    }




    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void dupDifferentOrdersOp(SetupState state) throws IOException {
        state.arrayUnordered.assign(state.arrayOrderedF);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void dupSameOrdersOp(SetupState state) throws IOException {
        state.arrayUnordered.assign(state.arrayOrderedC);
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void pairwiseOp1(SetupState state) throws IOException {
        state.array1.addiRowVector(state.array2);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void broadcastOp2(SetupState state) throws IOException {
        state.array.addiRowVector(state.arrayRow);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void reduceOp1(SetupState state) throws IOException {
        state.array.sum(0);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void reduceOp2(SetupState state) throws IOException {
        state.array.sumNumber().floatValue();
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void scalarOp1(SetupState state) throws IOException {
        state.array2.addi(0.5f);
    }


}
