package com.example.neanderthal;

import clojure.java.api.Clojure;
import clojure.lang.IFn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparison_8192x8192 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 8192;
        INDArray m1;
        INDArray m2;
        INDArray r;
        IFn mm;
        IFn fmap;
        IFn fge;
        Object n_m1;
        Object n_m2;
        Object n_r;

        @Setup(Level.Iteration)
        public void doSetup(){
            IFn require = Clojure.var("clojure.core", "require");
            IFn deref = Clojure.var("clojure.core", "deref");
            require.invoke(Clojure.read("uncomplicate.neanderthal.core"));
            require.invoke(Clojure.read("uncomplicate.neanderthal.native"));
            require.invoke(Clojure.read("uncomplicate.fluokitten.core"));

            fmap = Clojure.var("uncomplicate.fluokitten.core", "fmap!");
            mm = Clojure.var("uncomplicate.neanderthal.core", "mm!");
            fge = Clojure.var("uncomplicate.neanderthal.native", "fge");

            IFn random = (IFn) deref.invoke(Clojure.var("clojure.core", "eval").invoke(Clojure.read(
                    "(let [splittable-random (java.util.SplittableRandom.)]\n" +
                            "  (defn random ^double [^double _]\n" +
                            "    (.nextDouble ^java.util.SplittableRandom splittable-random)))"
            )));

            n_m1 = fmap.invoke(random, fge.invoke(size, size));
            n_m2 = fmap.invoke(random, fge.invoke(size, size));
            n_r = fge.invoke(size, size);

            m1 = Nd4j.rand(size, size);
            m2 = Nd4j.rand(m1.shape());
            r = Nd4j.createUninitialized(m1.shape(), 'f');
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_gemm(SetupState state) {
        Nd4j.gemm(state.m1, state.m2, state.r, false, false, 1.0, 0.0);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void neanderthal_mm(SetupState state) {
        state.mm.invoke(1.0f, state.n_m1, state.n_m2, 0.0f, state.n_r);
    }

}
