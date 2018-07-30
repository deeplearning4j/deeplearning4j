/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * This set of test launches different ops in different order, to check for possible data corruption cases
 *
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class CrashTest extends BaseNd4jTest {
    public CrashTest(Nd4jBackend backend) {
        super(backend);
    }

    private static final int ITERATIONS = 10;
    private static final boolean[] paramsA = new boolean[] {true, false};
    private static final boolean[] paramsB = new boolean[] {true, false};


    /**
     * tensorAlongDimension() produces shapeInfo without EWS defined
     */
    @Test
    public void testNonEWSViews1() {
        log.debug("non-EWS 1");
        INDArray x = Nd4j.create(64, 1024, 64);
        INDArray y = Nd4j.create(64, 64, 1024);

        for (int i = 0; i < ITERATIONS; i++) {
            int slice = RandomUtils.nextInt(0, (int) x.size(0));
            op(x.tensorAlongDimension(slice, 1, 2), y.tensorAlongDimension(slice, 1, 2), i);
        }
    }

    @Test
    public void testNonEWSViews2() {
        log.debug("non-EWS 2");
        INDArray x = Nd4j.create(new int[] {64, 1024, 64}, 'f');
        INDArray y = Nd4j.create(new int[] {64, 64, 1024}, 'f');

        for (int i = 0; i < ITERATIONS; i++) {
            int slice = RandomUtils.nextInt(0, (int) x.size(0));
            op(x.tensorAlongDimension(slice, 1, 2), y.tensorAlongDimension(slice, 1, 2), i);
        }
    }

    /**
     * slice() produces shapeInfo with EWS being 1 in our case
     */
    @Test
    public void testEWSViews1() {
        log.debug("EWS 1");
        INDArray x = Nd4j.create(64, 1024, 64);
        INDArray y = Nd4j.create(64, 64, 1024);

        for (int i = 0; i < ITERATIONS; i++) {
            // FIXME: int cast
            int slice = RandomUtils.nextInt(0, (int) x.shape()[0]);
            op(x.slice(slice), y.slice(slice), i);
        }
    }

    @Test
    public void testEWSViews2() {
        log.debug("EWS 2");
        INDArray x = Nd4j.create(new int[] {96, 1024, 64}, 'f');
        INDArray y = Nd4j.create(new int[] {96, 64, 1024}, 'f');

        for (int i = 0; i < 1; i++) {
            int slice = 0; //RandomUtils.nextInt(0, x.shape()[0]);
            op(x.slice(slice), y.slice(slice), i);
        }
    }

    protected void op(INDArray x, INDArray y, int i) {
        // broadcast along row & column
        INDArray row = Nd4j.ones(64);
        INDArray column = Nd4j.ones(1024, 1);

        x.addiRowVector(row);
        x.addiColumnVector(column);

        // casual scalar
        x.addi(i * 2);

        // reduction along all dimensions
        float sum = x.sumNumber().floatValue();

        // index reduction
        Nd4j.getExecutioner().exec(new IMax(x), Integer.MAX_VALUE);

        // casual transform
        Nd4j.getExecutioner().exec(new Sqrt(x, x));

        //  dup
        INDArray x1 = x.dup(x.ordering());
        INDArray x2 = x.dup(x.ordering());
        INDArray x3 = x.dup('c');
        INDArray x4 = x.dup('f');


        // vstack && hstack
        INDArray vstack = Nd4j.vstack(x, x1, x2, x3, x4);

        INDArray hstack = Nd4j.hstack(x, x1, x2, x3, x4);

        // reduce3 call
        Nd4j.getExecutioner().exec(new ManhattanDistance(x, x2));


        // flatten call
        INDArray flat = Nd4j.toFlattened(x, x1, x2, x3, x4);


        // reduction along dimension: row & column
        INDArray max_0 = x.max(0);
        INDArray max_1 = x.max(1);


        // index reduction along dimension: row & column
        INDArray imax_0 = Nd4j.argMax(x, 0);
        INDArray imax_1 = Nd4j.argMax(x, 1);


        // logisoftmax, softmax & softmax derivative
        Nd4j.getExecutioner().exec(new OldSoftMax(x));
        Nd4j.getExecutioner().exec(new SoftMaxDerivative(x));
        Nd4j.getExecutioner().exec(new LogSoftMax(x));


        // BooleanIndexing
        BooleanIndexing.replaceWhere(x, 5f, Conditions.lessThan(8f));

        // assing on view
        BooleanIndexing.assignIf(x, x1, Conditions.greaterThan(-1000000000f));

        // std var along all dimensions
        float std = x.stdNumber().floatValue();

        // std var along row & col
        INDArray xStd_0 = x.std(0);
        INDArray xStd_1 = x.std(1);

        // blas call
        float dot = (float) Nd4j.getBlasWrapper().dot(x, x1);

        // mmul
        for (boolean tA : paramsA) {
            for (boolean tB : paramsB) {

                INDArray xT = tA ? x.dup() : x.dup().transpose();
                INDArray yT = tB ? y.dup() : y.dup().transpose();

                Nd4j.gemm(xT, yT, tA, tB);
            }
        }

        // specially for views, checking here without dup and rollover
        Nd4j.gemm(x, y, false, false);

        log.debug("Iteration passed: " + i);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
