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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression tests for the P6 RWKV WKV recurrence kernels (rwkv_wkv6 / rwkv_wkv7).
 * These are new native kernels validated against pure-Java double-precision
 * references implementing the same documented recurrence (self-consistency;
 * exact ggml parity is not asserted — the ops are unreferenced and there is no
 * golden without a llamacpp build).
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestRwkvWkvOps 2>&1 | tee /tmp/test-rwkv.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestRwkvWkvOps {

    private static final int B = 2, T = 4, H = 2, S = 3;

    private static double get(INDArray a, int b, int t, int h, int s) { return a.getDouble(b, t, h, s); }

    @Test
    public void testRwkvWkv6AgainstReference() {
        Nd4j.getRandom().setSeed(61);
        INDArray k = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray r = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray tf = Nd4j.rand(DataType.FLOAT, H, S).subi(0.5);
        INDArray td = Nd4j.rand(DataType.FLOAT, B, T, H, S).mul(0.5).add(0.3);  // decay in (0.3,0.8)
        INDArray state = Nd4j.rand(DataType.FLOAT, B, H, S, S).subi(0.5);

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("rwkv_wkv6")
                .addInputs(k, v, r, tf, td, state).build())[0];
        assertArrayEquals(new long[]{B, T, H, S}, out.shape());

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                double[][] ss = new double[S][S];
                for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) ss[i][j] = state.getDouble(b, h, i, j);
                for (int t = 0; t < T; t++) {
                    double[] y = new double[S];
                    for (int i = 0; i < S; i++) {
                        double ki = get(k, b, t, h, i), ri = get(r, b, t, h, i);
                        double tfi = tf.getDouble(h, i), tdi = get(td, b, t, h, i);
                        for (int j = 0; j < S; j++) {
                            double kv = ki * get(v, b, t, h, j);
                            y[j] += ri * (tfi * kv + ss[i][j]);
                            ss[i][j] = tdi * ss[i][j] + kv;
                        }
                    }
                    for (int j = 0; j < S; j++)
                        assertEquals(y[j], out.getDouble(b, t, h, j), 1e-4, "wkv6 [" + b + "," + t + "," + h + "," + j + "]");
                }
            }
        }
    }

    @Test
    public void testRwkvWkv7AgainstReference() {
        Nd4j.getRandom().setSeed(62);
        INDArray r = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray w = Nd4j.rand(DataType.FLOAT, B, T, H, S).mul(0.5).add(0.3);
        INDArray k = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray a = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray bb = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray state = Nd4j.rand(DataType.FLOAT, B, H, S, S).subi(0.5);

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("rwkv_wkv7")
                .addInputs(r, w, k, v, a, bb, state).build())[0];
        assertArrayEquals(new long[]{B, T, H, S}, out.shape());

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                double[][] ss = new double[S][S];
                for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) ss[i][j] = state.getDouble(b, h, i, j);
                for (int t = 0; t < T; t++) {
                    double[] sa = new double[S];
                    for (int i = 0; i < S; i++)
                        for (int j = 0; j < S; j++) sa[j] += get(a, b, t, h, i) * ss[i][j];
                    double[] y = new double[S];
                    for (int i = 0; i < S; i++) {
                        double ki = get(k, b, t, h, i), wi = get(w, b, t, h, i);
                        double bi = get(bb, b, t, h, i), ri = get(r, b, t, h, i);
                        for (int j = 0; j < S; j++) {
                            ss[i][j] = wi * ss[i][j] + ki * get(v, b, t, h, j) + bi * sa[j];
                            y[j] += ri * ss[i][j];
                        }
                    }
                    for (int j = 0; j < S; j++)
                        assertEquals(y[j], out.getDouble(b, t, h, j), 1e-4, "wkv7 [" + b + "," + t + "," + h + "," + j + "]");
                }
            }
        }
    }

    @Test
    public void testWkv6ZeroStateSingleToken() {
        // T=1, zero state: out[j] = sum_i r[i]*tf[i]*k[i]*v[j]
        INDArray k = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        INDArray r = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        INDArray tf = Nd4j.rand(DataType.FLOAT, 1, S).subi(0.5);
        INDArray td = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S);
        INDArray state = Nd4j.zeros(DataType.FLOAT, 1, 1, S, S);

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("rwkv_wkv6")
                .addInputs(k, v, r, tf, td, state).build())[0];

        for (int j = 0; j < S; j++) {
            double expected = 0;
            for (int i = 0; i < S; i++)
                expected += r.getDouble(0, 0, 0, i) * tf.getDouble(0, i) * k.getDouble(0, 0, 0, i) * v.getDouble(0, 0, 0, j);
            assertEquals(expected, out.getDouble(0, 0, 0, j), 1e-5, "wkv6 zero-state token " + j);
        }
    }
}
