package org.nd4j;

import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.All)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 0, time = 1)
@Measurement(iterations = 5, time = 1)
public class SmallNDArrays {
    private INDArray arrayOne;
    private INDArray arrayTwo;
    @Setup(Level.Trial)
    @Fork(value = 0, warmups = 1)
    public void setup() {
        System.out.println("Setting up PID: " + ProcessHandle.current().pid());
        arrayOne = Nd4j.ones(200);
        arrayTwo = Nd4j.ones(200);
    }

    @Benchmark
    public void add() {
        arrayOne.addi(arrayTwo);
    }


}

