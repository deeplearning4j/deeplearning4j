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
 * Regression tests for the P7 gated_linear_attn kernel, validated against a
 * pure-Java double-precision reference of the same GLA recurrence (gated and
 * ungated), starting from zero state.
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestGatedLinearAttn 2>&1 | tee /tmp/test-gla.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestGatedLinearAttn {

    private static final int B = 2, T = 4, H = 2, S = 3;

    private static double get(INDArray a, int b, int t, int h, int s) { return a.getDouble(b, t, h, s); }

    /** Reference GLA recurrence, zero initial state. gate may be null. */
    private static void checkAgainstReference(INDArray q, INDArray k, INDArray v, INDArray gate,
                                              double scale, INDArray out) {
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                double[][] ss = new double[S][S];  // zero state
                for (int t = 0; t < T; t++) {
                    double[] o = new double[S];
                    for (int i = 0; i < S; i++) {
                        double gi = gate != null ? get(gate, b, t, h, i) : 1.0;
                        double ki = get(k, b, t, h, i), qi = get(q, b, t, h, i);
                        for (int j = 0; j < S; j++) {
                            ss[i][j] = gi * ss[i][j] + ki * get(v, b, t, h, j);
                            o[j] += qi * ss[i][j];
                        }
                    }
                    for (int j = 0; j < S; j++)
                        assertEquals(o[j] * scale, out.getDouble(b, t, h, j), 1e-4,
                                "gla [" + b + "," + t + "," + h + "," + j + "]");
                }
            }
        }
    }

    @Test
    public void testGatedLinearAttnWithGate() {
        Nd4j.getRandom().setSeed(71);
        INDArray q = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray k = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray gate = Nd4j.rand(DataType.FLOAT, B, T, H, S).mul(0.4).add(0.4);  // decay in (0.4,0.8)
        double scale = 0.5;

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("gated_linear_attn")
                .addInputs(q, k, v, gate).addFloatingPointArguments(scale).build())[0];
        assertArrayEquals(new long[]{B, T, H, S}, out.shape());
        checkAgainstReference(q, k, v, gate, scale, out);
    }

    @Test
    public void testUngatedDefaultScale() {
        Nd4j.getRandom().setSeed(72);
        INDArray q = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray k = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, B, T, H, S).subi(0.5);
        double scale = 1.0 / Math.sqrt(S);  // default

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("gated_linear_attn")
                .addInputs(q, k, v).build())[0];  // no gate, no scale arg → default
        assertArrayEquals(new long[]{B, T, H, S}, out.shape());
        checkAgainstReference(q, k, v, null, scale, out);
    }

    @Test
    public void testSingleTokenClosedForm() {
        // T=1, zero state: state[i,j] = k[i]*v[j]; out[j] = scale * sum_i q[i]*k[i]*v[j]
        Nd4j.getRandom().setSeed(73);
        INDArray q = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        INDArray k = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        INDArray v = Nd4j.rand(DataType.FLOAT, 1, 1, 1, S).subi(0.5);
        double scale = 0.7;

        INDArray out = Nd4j.exec(DynamicCustomOp.builder("gated_linear_attn")
                .addInputs(q, k, v).addFloatingPointArguments(scale).build())[0];

        for (int j = 0; j < S; j++) {
            double qk = 0;
            for (int i = 0; i < S; i++) qk += q.getDouble(0, 0, 0, i) * k.getDouble(0, 0, 0, i);
            double expected = scale * qk * v.getDouble(0, 0, 0, j);
            assertEquals(expected, out.getDouble(0, 0, 0, j), 1e-5, "gla single-token " + j);
        }
    }
}
