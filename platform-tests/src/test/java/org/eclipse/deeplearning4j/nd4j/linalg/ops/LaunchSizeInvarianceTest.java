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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Ops at sizes past where their CUDA launches broke equal the same ops on small pieces of the same data. A commit
 * flipped the CUDA launch-dimension producers to (blocks, threads) without their call sites, which kept launching
 * (y, x): the block count became the thread count, so an op failed (invalid configuration, or dynamic shared memory
 * past its limit) once its length needed more than 1024 blocks of the intended size.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class LaunchSizeInvarianceTest extends BaseNd4jTestWithBackends {

    private static INDArray[] exec(String op, INDArray[] inputs, double[] tArgs, long[] iArgs) {
        DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder(op).addInputs(inputs);
        if (tArgs.length > 0) {
            List<Double> t = new ArrayList<>();
            for (double d : tArgs)
                t.add(d);
            builder.addFloatingPointArguments(t);
        }
        if (iArgs.length > 0)
            builder.addIntegerArguments(iArgs);
        return Nd4j.exec(builder.build());
    }

    /**
     * An elementwise op over vectors of one length: the outputs over the whole vectors equal the outputs over each
     * piece of {@code chunk} elements.
     */
    private static void assertElementwiseChunks(String op, INDArray[] inputs, double[] tArgs, long[] iArgs,
                                                long chunk) {
        INDArray[] full = exec(op, inputs, tArgs, iArgs);
        long n = inputs[0].length();
        for (long start = 0; start < n; start += chunk) {
            long end = Math.min(n, start + chunk);
            INDArray[] part = new INDArray[inputs.length];
            for (int i = 0; i < inputs.length; i++)
                part[i] = inputs[i].get(NDArrayIndex.interval(start, end)).dup();
            INDArray[] pieces = exec(op, part, tArgs, iArgs);
            for (int o = 0; o < full.length; o++)
                assertEquals(pieces[o], full[o].get(NDArrayIndex.interval(start, end)),
                        op + " output " + o + ", elements " + start + " to " + end + " of " + n);
        }
    }

    /** Uniform values in [lo, hi) from a fixed seed. */
    private static INDArray uniform(long n, double lo, double hi, long seed) {
        Nd4j.getRandom().setSeed(seed);
        return Nd4j.rand(DataType.FLOAT, n).muli(hi - lo).addi(lo);
    }

    // name, state inputs, tArgs, takes an iteration
    private static final Object[][] UPDATERS = {
            {"adabelief_updater", 2, new double[]{1e-3, 0.9, 0.999, 1e-8}, true},
            {"ada_delta_updater", 2, new double[]{0.95, 1e-6}, false},
            {"ada_grad_updater", 1, new double[]{1e-2, 1e-6}, false},
            {"ada_max_updater", 2, new double[]{1e-3, 0.9, 0.999, 1e-8}, true},
            {"adam_updater", 2, new double[]{1e-3, 0.9, 0.999, 1e-8}, true},
            {"ams_grad_updater", 3, new double[]{1e-3, 0.9, 0.999, 1e-8}, true},
            {"nadam_updater", 2, new double[]{1e-3, 0.9, 0.999, 1e-8}, true},
            {"nesterovs_updater", 1, new double[]{1e-2, 0.9}, false},
            {"rms_prop_updater", 1, new double[]{1e-2, 0.95, 1e-8}, false}};

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void updatersPastAMillionElements(Nd4jBackend backend) {
        // 1025 blocks of 256 threads from 262145 elements on: a 1024 x 1024 weight's update threw
        long n = (1L << 20) + 3;
        for (Object[] u : UPDATERS) {
            int states = (Integer) u[1];
            INDArray[] inputs = new INDArray[1 + states];
            inputs[0] = uniform(n, -1, 1, 11);
            for (int s = 0; s < states; s++)
                inputs[1 + s] = uniform(n, 0.01, 1, 12 + s);
            long[] iArgs = (Boolean) u[3] ? new long[]{3} : new long[0];
            assertElementwiseChunks((String) u[0], inputs, (double[]) u[2], iArgs, 1 << 16);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void mergeOpsPastHalfAMillionElements(Nd4jBackend backend) {
        // 1025 blocks of 512 threads from 524289 elements on
        long n = (1L << 19) + 5;
        INDArray[] inputs = {uniform(n, -1, 1, 21), uniform(n, -1, 1, 22), uniform(n, -1, 1, 23)};
        for (String op : new String[]{"mergeadd", "mergemax", "mergeavg"})
            assertElementwiseChunks(op, inputs, new double[0], new long[0], 1 << 16);
        assertElementwiseChunks("mergemaxindex", inputs, new double[0], new long[]{DataType.INT32.toInt()}, 1 << 16);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gammaFunctionsPastHalfAMillionElements(Nd4jBackend backend) {
        long n = (1L << 19) + 5;
        INDArray x = uniform(n, 0.1, 20, 31);
        assertElementwiseChunks("digamma", new INDArray[]{x}, new double[0], new long[0], 1 << 16);
        INDArray order = Nd4j.valueArrayOf(new long[]{n}, 2.0, DataType.FLOAT);
        assertElementwiseChunks("polygamma", new INDArray[]{order, x}, new double[0], new long[0], 1 << 16);
    }

    /** eye and the triangle ops at sizes whose per-thread coordinates overflowed the launch's shared memory. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void identityAndTrianglesPastTheirSharedMemory(Nd4jBackend backend) {
        for (int n : new int[]{11, 12, 187, 513}) {
            INDArray eye = Nd4j.exec(DynamicCustomOp.builder("eye").addIntegerArguments(n, n).build())[0];
            INDArray square = uniform((long) n * n, -1, 1, n).reshape(n, n);
            INDArray triu = Nd4j.exec(DynamicCustomOp.builder("triu").addInputs(square).addIntegerArguments(1).build())[0];
            double[][] values = square.toDoubleMatrix();
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    assertEquals(r == c ? 1.0 : 0.0, eye.getDouble(r, c), 0.0, "eye " + n + " at " + r + ", " + c);
                    assertEquals(c - r >= 1 ? values[r][c] : 0.0, triu.getDouble(r, c), 0.0,
                            "triu(1) " + n + " at " + r + ", " + c);
                }
            }
        }
    }
}
