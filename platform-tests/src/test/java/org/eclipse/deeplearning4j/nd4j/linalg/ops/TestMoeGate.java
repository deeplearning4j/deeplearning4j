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
import org.nd4j.linalg.api.ops.impl.transforms.custom.MoeGate;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the native moe_gate op (top-K MoE router with
 * load-balancing auxiliary loss).
 *
 * <p>This exercises the exact production failure path: constructing MoeGate with
 * no pre-allocated outputs forces output-shape calculation through the C++ op
 * descriptor, which previously failed with
 * "Could not find descriptor for op: moe_gate" (the op only existed as a
 * llamacpp platform override).</p>
 *
 * <h2>Reference strategy</h2>
 * A pure-Java double-precision re-implementation computes
 * logits = hidden @ gateW, row softmax, top-K selection, renormalized weights,
 * and the aux loss coeff * E * sum_e(meanProb_e * routedFrac_e). Op outputs are
 * compared against it with tight float tolerances.
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestMoeGate 2>&1 | tee /tmp/test-moe-gate.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestMoeGate {

    private static final float DEFAULT_AUX_COEFF = 0.01f;

    private static double[][] toMatrix(INDArray arr) {
        double[][] out = new double[(int) arr.size(0)][(int) arr.size(1)];
        for (int i = 0; i < out.length; i++)
            for (int j = 0; j < out[0].length; j++)
                out[i][j] = arr.getDouble(i, j);
        return out;
    }

    /** Pure-Java reference: returns {indices [T][K], weights [T][K], auxLoss}. */
    private static Object[] reference(double[][] hidden, double[][] gateW, int numExperts, int topK,
                                      double auxCoeff) {
        int tokens = hidden.length, hiddenDim = hidden[0].length;
        double[][] probs = new double[tokens][numExperts];
        for (int t = 0; t < tokens; t++) {
            double[] logits = new double[numExperts];
            double max = Double.NEGATIVE_INFINITY;
            for (int e = 0; e < numExperts; e++) {
                double v = 0;
                for (int h = 0; h < hiddenDim; h++) v += hidden[t][h] * gateW[h][e];
                logits[e] = v;
                max = Math.max(max, v);
            }
            double sum = 0;
            for (int e = 0; e < numExperts; e++) {
                probs[t][e] = Math.exp(logits[e] - max);
                sum += probs[t][e];
            }
            for (int e = 0; e < numExperts; e++) probs[t][e] /= sum;
        }

        long[][] indices = new long[tokens][topK];
        double[][] weights = new double[tokens][topK];
        long[] routedCount = new long[numExperts];
        for (int t = 0; t < tokens; t++) {
            Integer[] order = new Integer[numExperts];
            for (int e = 0; e < numExperts; e++) order[e] = e;
            final double[] row = probs[t];
            Arrays.sort(order, (a, b) -> Double.compare(row[b], row[a]));
            double selSum = 0;
            for (int k = 0; k < topK; k++) selSum += row[order[k]];
            for (int k = 0; k < topK; k++) {
                indices[t][k] = order[k];
                weights[t][k] = row[order[k]] / selSum;
                routedCount[order[k]]++;
            }
        }

        double balance = 0;
        for (int e = 0; e < numExperts; e++) {
            double meanProb = 0;
            for (int t = 0; t < tokens; t++) meanProb += probs[t][e];
            meanProb /= tokens;
            double frac = routedCount[e] / (double) (tokens * (long) topK);
            balance += meanProb * frac;
        }
        double auxLoss = auxCoeff * numExperts * balance;
        return new Object[]{indices, weights, auxLoss};
    }

    @Test
    public void testOutputShapesAndTypesViaShapeCalc() {
        // No pre-allocated outputs: shape calculation must go through the C++ descriptor —
        // this is the exact call path that failed in production.
        Nd4j.getRandom().setSeed(42);
        int tokens = 5, hiddenDim = 8, numExperts = 4, topK = 2;
        INDArray hidden = Nd4j.rand(DataType.FLOAT, tokens, hiddenDim).subi(0.5);
        INDArray gateW = Nd4j.rand(DataType.FLOAT, hiddenDim, numExperts).subi(0.5);

        INDArray[] out = Nd4j.exec(new MoeGate(hidden, gateW, numExperts, topK));

        assertEquals(3, out.length, "moe_gate must produce 3 outputs");
        assertEquals(DataType.INT64, out[0].dataType(), "expertIndices must be INT64");
        assertArrayEquals(new long[]{tokens, topK}, out[0].shape());
        assertEquals(hidden.dataType(), out[1].dataType());
        assertArrayEquals(new long[]{tokens, topK}, out[1].shape());
        assertEquals(1, out[2].length(), "auxLoss must be a single-element array");

        for (int t = 0; t < tokens; t++) {
            for (int k = 0; k < topK; k++) {
                long idx = out[0].getLong(t, k);
                assertTrue(idx >= 0 && idx < numExperts,
                        "expert index out of range at [" + t + "," + k + "]: " + idx);
            }
        }
    }

    @Test
    public void testAgainstJavaReference() {
        Nd4j.getRandom().setSeed(12345);
        int tokens = 7, hiddenDim = 16, numExperts = 6, topK = 2;
        INDArray hidden = Nd4j.rand(DataType.FLOAT, tokens, hiddenDim).subi(0.5);
        INDArray gateW = Nd4j.rand(DataType.FLOAT, hiddenDim, numExperts).subi(0.5);

        INDArray[] out = Nd4j.exec(new MoeGate(hidden, gateW, numExperts, topK));

        Object[] ref = reference(toMatrix(hidden), toMatrix(gateW), numExperts, topK, DEFAULT_AUX_COEFF);
        long[][] expIdx = (long[][]) ref[0];
        double[][] expW = (double[][]) ref[1];
        double expLoss = (double) ref[2];

        for (int t = 0; t < tokens; t++) {
            double rowSum = 0;
            for (int k = 0; k < topK; k++) {
                assertEquals(expIdx[t][k], out[0].getLong(t, k),
                        "expert index mismatch at [" + t + "," + k + "]");
                assertEquals(expW[t][k], out[1].getDouble(t, k), 1e-4,
                        "gate weight mismatch at [" + t + "," + k + "]");
                rowSum += out[1].getDouble(t, k);
            }
            assertEquals(1.0, rowSum, 1e-5, "gate weights of token " + t + " must sum to 1");
        }
        assertEquals(expLoss, out[2].getDouble(0), 1e-5, "aux loss mismatch");
    }

    @Test
    public void testTopK1IsArgmax() {
        Nd4j.getRandom().setSeed(7);
        int tokens = 4, hiddenDim = 8, numExperts = 5;
        INDArray hidden = Nd4j.rand(DataType.FLOAT, tokens, hiddenDim).subi(0.5);
        INDArray gateW = Nd4j.rand(DataType.FLOAT, hiddenDim, numExperts).subi(0.5);

        INDArray[] out = Nd4j.exec(new MoeGate(hidden, gateW, numExperts, 1));

        INDArray logits = hidden.mmul(gateW);
        for (int t = 0; t < tokens; t++) {
            assertEquals(logits.getRow(t).argMax().getLong(0), out[0].getLong(t, 0),
                    "top-1 must equal argmax for token " + t);
            assertEquals(1.0, out[1].getDouble(t, 0), 1e-6,
                    "top-1 renormalized weight must be exactly 1");
        }
    }
}
