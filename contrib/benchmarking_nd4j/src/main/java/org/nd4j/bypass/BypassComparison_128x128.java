
/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.bypass;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.openblas.global.openblas;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

public class BypassComparison_128x128 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 128;
        public INDArray m1 = Nd4j.ones(size, size);
        public INDArray m2 = Nd4j.ones(m1.shape());
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'f');


        public GemmParams params = new GemmParams(m1, m2, r);
        FloatPointer a = (FloatPointer) params.getA().data().addressPointer();
        FloatPointer b = (FloatPointer) params.getB().data().addressPointer();
        FloatPointer c = (FloatPointer) params.getC().data().addressPointer();

        int M = params.getM();
        int N = params.getN();
        int K = params.getK();
        int lda = params.getLda();
        int ldb = params.getLdb();
        int ldc = params.getLdc();


        public Level3 wrapper = Nd4j.getBlasWrapper().level3();
        public Method sgemm;

        @Setup(Level.Iteration)
        public void doSetup(){
            try {
                sgemm = wrapper.getClass().getDeclaredMethod("sgemm", char.class, char.class, char.class, int.class, int.class, int.class, float.class, INDArray.class,
                        int.class, INDArray.class, int.class, float.class, INDArray.class, int.class);
                sgemm.setAccessible(true);
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
        }
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void mmuli(SetupState state) {
        state.m1.mmuli(state.m2, state.r);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_gemm(SetupState state) {
        Nd4j.gemm(state.m1, state.m2, state.r, false, false, 1.0, 0.0);
    }

    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void openblas_cblas_gemm(SetupState state, Blackhole bh) {
        openblas.cblas_sgemm(102,111, 111, state.M, state.N, state.K, 1.0f, state.a, state.lda, state.b, state.ldb, 0.0f, state.c, state.ldc);
    }


    @Benchmark @BenchmarkMode(Mode.AverageTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void level3_sgemm(SetupState state, Blackhole bh) {
        final GemmParams params = state.params;
        try {
            state.sgemm.invoke(state.wrapper, params.getA().ordering(), params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(), (float)1.0, params.getA(), params.getLda(), params.getB(), params.getLdb(), (float)0.0, params.getC(), params.getLdc());
        } catch (IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
        }
    }
}
