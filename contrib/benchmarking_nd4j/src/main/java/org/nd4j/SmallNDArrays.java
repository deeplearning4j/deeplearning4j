package org.nd4j;

import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 0, time = 3)
@Measurement(iterations = 5, time = 3, timeUnit = TimeUnit.NANOSECONDS)
public class SmallNDArrays {
    private INDArray arrayOne;
    private INDArray arrayTwo;
    @Setup(Level.Trial)
    @Fork(value = 3, warmups = 3)
    public void setup() throws Exception {
        System.out.println("Setting up PID: " + ProcessHandle.current().pid());
        arrayOne = Nd4j.ones(200);
        arrayTwo = Nd4j.ones(200);
    }

    @Benchmark
    public void add() {
        arrayOne.addi(arrayTwo);
    }


}

