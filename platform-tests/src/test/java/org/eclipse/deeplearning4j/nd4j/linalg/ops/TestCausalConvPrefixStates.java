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

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1dWithPrefix;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Prefix-history capture contract for causal_conv1d_with_prefix.
 *
 * For every live prefix m, checkpoint slot m-1 must equal the final state of an
 * independent legacy execution at actualLen=m from the SAME nonzero initial state,
 * and ordinary outputs must be identical. Together with the GDN prefix fixture
 * this is the oracle that lets the MTP controller select prefix[consumed-1]
 * instead of re-executing the target for the consumed prefix.
 */
public class TestCausalConvPrefixStates {

    private static final int B = 1;
    private static final int D = 4;
    private static final int KC = 4; // convolution width; history length KC-1 = 3

    private static INDArray[] deterministicInputs(int l) {
        INDArray x = Nd4j.linspace(1, B * l * D, B * l * D, DataType.FLOAT)
                .reshape(B, l, D).muli(0.1f);
        INDArray weight = Nd4j.linspace(1, D * KC, D * KC, DataType.FLOAT)
                .reshape(D, KC).muli(0.01f);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, D);
        return new INDArray[]{x, weight, bias};
    }

    private static INDArray nonzeroHistory() {
        INDArray s = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        float value = 0.25f;
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                for (int kk = 0; kk < KC - 1; kk++) {
                    s.putScalar(new int[]{b, d, kk}, value);
                    value = value * -1.5f + 0.125f;
                }
            }
        }
        return s;
    }

    private static void assertFinite(INDArray a, String name) {
        assertTrue(!a.isNaN().any(), name + " contains NaN");
        assertTrue(!a.isInfinite().any(), name + " contains Inf");
    }

    /** history_m element [b, d, kk] computed directly from the declared oracle. */
    private static float oracleHistory(INDArray stateIn, INDArray x, int b, int d, int kk, int m) {
        int srcT = m - (KC - 1) + (int) kk;
        if (srcT >= 0) {
            return x.getFloat(b, srcT, d);
        }
        int stateIdx = (KC - 1) + srcT;
        if (stateIdx >= 0) {
            return stateIn.getFloat(b, d, stateIdx);
        }
        return 0.0f;
    }

    @Test
    public void testPrefixSlotsMatchOracleAndLegacy() {
        for (int l : new int[]{2, 5, 8}) {
            INDArray[] in = deterministicInputs(l);
            INDArray x = in[0];
            INDArray weight = in[1];
            INDArray bias = in[2];
            INDArray stateIn = nonzeroHistory();
            INDArray stateInBackup = stateIn.dup();

            INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
            INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
            INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);

            CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                    x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
            op.addOutputArgument(output, stateOut, prefix);
            Nd4j.getExecutioner().exec(op);
            Nd4j.getExecutioner().commit();

            assertFinite(output, "output L=" + l);
            assertFinite(stateOut, "stateOut L=" + l);
            assertFinite(prefix, "prefix L=" + l);

            // Every prefix slot against the direct oracle.
            for (int m = 1; m <= l; m++) {
                for (int kk = 0; kk < KC - 1; kk++) {
                    assertEquals(oracleHistory(stateInBackup, x, 0, 0, kk, m),
                            prefix.getFloat(m - 1, 0, 0, kk), 0.0f,
                            "L=" + l + " prefix[" + (m - 1) + "] kk=" + kk);
                    assertEquals(oracleHistory(stateInBackup, x, 0, D - 1, kk, m),
                            prefix.getFloat(m - 1, 0, D - 1, kk), 0.0f,
                            "L=" + l + " last-channel prefix[" + (m - 1) + "] kk=" + kk);
                }
            }

            // Final state equals legacy execution at actualLen = l.
            INDArray legacyState = runLegacyState(x, weight, bias, stateInBackup, l);
            assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                    "L=" + l + " final state differs from legacy");
            assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                    "stateIn was mutated, L=" + l);
        }
    }

    @Test
    public void testShortenedActualLenMatchesLegacy() {
        int l = 8;
        int active = 5;
        INDArray[] in = deterministicInputs(l);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);

        CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) active));
        op.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();

        for (int m = 1; m <= active; m++) {
            for (int kk = 0; kk < KC - 1; kk++) {
                assertEquals(oracleHistory(stateInBackup, x, 0, 1, kk, m),
                        prefix.getFloat(m - 1, 0, 1, kk), 0.0f,
                        "active=" + active + " prefix[" + (m - 1) + "] kk=" + kk);
            }
        }
        // Slots beyond active are not written by this invocation; assert final state
        // equals the legacy shortened run.
        INDArray legacyState = runLegacyState(x, weight, bias, stateInBackup, active);
        assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                "active=" + active + " final state differs from legacy");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                "stateIn was mutated");
    }

    private static INDArray runLegacyState(INDArray x, INDArray weight, INDArray bias,
                                           INDArray stateIn, int actualLen) {
        INDArray output = Nd4j.create(DataType.FLOAT, B, x.size(1), D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        CausalConv1d op = new CausalConv1d(x, weight, bias, stateIn,
                Nd4j.scalar(DataType.INT64, (long) actualLen), 0, 0);
        op.addOutputArgument(output, stateOut);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        return stateOut;
    }
}
