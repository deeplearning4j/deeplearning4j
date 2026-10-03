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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Closed-form gradient checks of {@code dot_product_attention_v2_bp} (rank 3) with query and value masks.
 *
 * <p>The contract these tests pin down is Keras' Attention layer, which the forward op implements:
 * <pre>
 *   logits  = scale * Q K^T, plus -1e9 where valueMask == 0
 *   weights = softmax(logits)               (last axis; returned unmasked as output 1)
 *   out     = (weights @ V) * queryMask[..., None]
 * </pre>
 * A masked query's output row is zero, and the weights it attended with stay visible. (The op once multiplied the
 * returned weights by the query mask instead, left the output unmasked, and then read the masked weights back in the
 * backward as if they were the softmax: every gradient of a masked query's batch was zero.)
 *
 * <p>Math of the expected gradients, for upstream gradient E of {@code out}, E' = E * queryMask[..., None], W the
 * softmax weights and {@code dW = E' V^T}:
 * <pre>
 *   dV = W^T E'
 *   dLogits = W * (dW - rowsum(dW * W))     (zero at masked values, where W == 0)
 *   dQ = scale * dLogits K
 *   dK = scale * dLogits^T Q
 * </pre>
 * The reference below evaluates these with plain loops, independent of libnd4j.
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.FULL_CI)
@DisplayName("DotProductAttentionV2 query/value mask gradients (closed form)")
public class TestDotProductAttentionV2MaskGradients extends BaseOpValidation {

    private static final int BATCH = 4;
    private static final int QUERY_LENGTH = 3;
    private static final int VALUE_LENGTH = 5;
    private static final int DIM = 4;
    // Exactly representable in float32: the fused (unmasked) path narrows the scale to float
    private static final double SCALE = 0.625;
    private static final double TOLERANCE = 1e-8;

    // queryMask [batch, Tq]: batch 0 is masked entirely, batch 3 partially
    private static final double[][] QUERY_MASK = {{0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
    // valueMask [batch, Tv]: every batch keeps at least one value
    private static final double[][] VALUE_MASK = {{1, 1, 0, 1, 1}, {1, 0, 1, 1, 0}, {0, 1, 1, 1, 1}, {1, 1, 1, 0, 1}};

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ========================= op level =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("query + value mask: dQ, dK, dV match the closed form")
    public void testBackwardWithQueryAndValueMask(Nd4jBackend backend) {
        checkAgainstClosedForm("query+value mask", QUERY_MASK, VALUE_MASK);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("query mask only: dQ, dK, dV match the closed form")
    public void testBackwardWithQueryMaskOnly(Nd4jBackend backend) {
        checkAgainstClosedForm("query mask only", QUERY_MASK, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("value mask only: dQ, dK, dV match the closed form")
    public void testBackwardWithValueMaskOnly(Nd4jBackend backend) {
        checkAgainstClosedForm("value mask only", null, VALUE_MASK);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("no masks: dQ, dK, dV match the closed form")
    public void testBackwardWithoutMasks(Nd4jBackend backend) {
        checkAgainstClosedForm("no masks", null, null);
    }

    // ========================= SameDiff level =========================

    /**
     * The failing scenario of {@code TestReductionOpValidation#testDotProductAttentionV2WithMask} with fixed data:
     * a norm1 loss over positive outputs (upstream gradient of ones), a query mask that masks a whole batch, and the
     * gradients taken through {@code DotProductAttentionV2#doDiff}.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("SameDiff gradients keep a fully query-masked batch")
    public void testSameDiffGradientsWithMaskedQueryBatch(Nd4jBackend backend) {
        double[][][] query = queries();
        double[][][] keys = keys();
        double[][][] values = values();
        double[][][] ones = filled(BATCH, QUERY_LENGTH, DIM, 1.0);

        SameDiff sd = SameDiff.create();
        SDVariable sdQuery = sd.var("q", Nd4j.createFromArray(query));
        SDVariable sdValues = sd.var("values", Nd4j.createFromArray(values));
        SDVariable sdKeys = sd.var("keys", Nd4j.createFromArray(keys));
        SDVariable sdQueryMask = sd.constant("queryMask", Nd4j.createFromArray(QUERY_MASK));
        SDVariable sdValueMask = sd.constant("valueMask", Nd4j.createFromArray(VALUE_MASK));

        SDVariable attention = sd.nn().dotProductAttentionV2(sdQuery, sdValues, sdKeys, sdQueryMask, sdValueMask,
                SCALE, 0.0, false, true);
        SDVariable loss = attention.norm1("loss");
        loss.markAsLoss();

        Map<String, INDArray> gradients = sd.calculateGradients(null, "q", "values", "keys");
        Expected expected = reference(query, keys, values, QUERY_MASK, VALUE_MASK, ones, SCALE);

        assertClose("dQ", expected.dQ, gradients.get("q"));
        assertClose("dValues", expected.dV, gradients.get("values"));
        assertClose("dKeys", expected.dK, gradients.get("keys"));
        sd.close();
    }

    // ========================= helpers =========================

    /**
     * Runs {@code dot_product_attention_v2} and {@code dot_product_attention_v2_bp} on the same inputs and compares
     * the forward output and the three gradients with the closed form.
     */
    private static void checkAgainstClosedForm(String label, double[][] queryMask, double[][] valueMask) {
        double[][][] query = queries();
        double[][][] keys = keys();
        double[][][] values = values();
        double[][][] upstream = upstreamGradient();

        INDArray queryArray = Nd4j.createFromArray(query);
        INDArray keysArray = Nd4j.createFromArray(keys);
        INDArray valuesArray = Nd4j.createFromArray(values);
        INDArray upstreamArray = Nd4j.createFromArray(upstream);

        DynamicCustomOp forward = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(queryArray, valuesArray, keysArray, maskOrEmpty(queryMask), maskOrEmpty(valueMask))
                .addFloatingPointArguments(SCALE, 0.0)      // T_ARG: scale, dropout
                .addBooleanArguments(false, true, true)     // B_ARG: useCausalMask, training, useFlashAttention
                .build();
        INDArray[] forwardOutputs = Nd4j.exec(forward);

        // Backward inputs: queries, values, keys, out, weights, logits, eps, dropoutMask, queryMask, valueMask
        DynamicCustomOp backward = DynamicCustomOp.builder("dot_product_attention_v2_bp")
                .addInputs(queryArray, valuesArray, keysArray, forwardOutputs[0], forwardOutputs[1],
                        forwardOutputs[2], upstreamArray, Nd4j.empty(DataType.DOUBLE), maskOrEmpty(queryMask),
                        maskOrEmpty(valueMask))
                .addFloatingPointArguments(SCALE, 0.0)      // T_ARG: scale, dropout
                .addBooleanArguments(false, true)           // B_ARG: useCausalMask, training
                .build();
        INDArray[] gradients = Nd4j.exec(backward);

        Expected expected = reference(query, keys, values, queryMask, valueMask, upstream, SCALE);

        // The query mask zeroes a masked query's output row and leaves the returned weights alone (see class comment)
        assertClose(label + " out", expected.out, forwardOutputs[0]);
        assertClose(label + " weights", expected.weights, forwardOutputs[1]);
        // Backward outputs are ordered like the forward inputs: queries, values, keys
        assertClose(label + " dQ", expected.dQ, gradients[0]);
        assertClose(label + " dV", expected.dV, gradients[1]);
        assertClose(label + " dK", expected.dK, gradients[2]);
    }

    private static INDArray maskOrEmpty(double[][] mask) {
        return mask == null ? Nd4j.empty(DataType.DOUBLE) : Nd4j.createFromArray(mask);
    }

    private static double[][][] queries() {
        return pattern(BATCH, QUERY_LENGTH, DIM, 0.37, 0.11, 0.0, 0.8);
    }

    private static double[][][] keys() {
        return pattern(BATCH, VALUE_LENGTH, DIM, 0.23, 0.5, 0.0, 0.9);
    }

    /** Strictly positive, so that a norm1 loss over the attention output has an upstream gradient of ones. */
    private static double[][][] values() {
        return pattern(BATCH, VALUE_LENGTH, DIM, 0.31, 1.0, 1.0, 0.4);
    }

    private static double[][][] upstreamGradient() {
        return pattern(BATCH, QUERY_LENGTH, DIM, 0.17, 0.3, 0.0, 1.0);
    }

    private static double[][][] pattern(int batch, int rows, int cols, double frequency, double phase,
                                        double offset, double amplitude) {
        double[][][] tensor = new double[batch][rows][cols];
        int n = 0;
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++, n++) {
                    tensor[b][r][c] = offset + amplitude * Math.sin(frequency * n + phase);
                }
            }
        }
        return tensor;
    }

    private static double[][][] filled(int batch, int rows, int cols, double value) {
        double[][][] tensor = new double[batch][rows][cols];
        for (int b = 0; b < batch; b++) {
            for (int r = 0; r < rows; r++) {
                Arrays.fill(tensor[b][r], value);
            }
        }
        return tensor;
    }

    private static void assertClose(String name, double[][][] expected, INDArray actual) {
        assertNotNull(actual, name + ": no array");
        assertEquals(expected.length, actual.size(0), name + ": batch size");
        assertEquals(expected[0].length, actual.size(1), name + ": rows");
        assertEquals(expected[0][0].length, actual.size(2), name + ": columns");
        for (int b = 0; b < expected.length; b++) {
            for (int r = 0; r < expected[b].length; r++) {
                for (int c = 0; c < expected[b][r].length; c++) {
                    assertEquals(expected[b][r][c], actual.getDouble(b, r, c), TOLERANCE,
                            name + "[" + b + "," + r + "," + c + "]");
                }
            }
        }
    }

    private static final class Expected {
        final double[][][] out;
        final double[][][] weights;
        final double[][][] dQ;
        final double[][][] dV;
        final double[][][] dK;

        Expected(double[][][] out, double[][][] weights, double[][][] dQ, double[][][] dV, double[][][] dK) {
            this.out = out;
            this.weights = weights;
            this.dQ = dQ;
            this.dV = dV;
            this.dK = dK;
        }
    }

    /**
     * Forward output, weights and gradients of {@code out = (softmax(scale * Q K^T masked) V) * queryMask[..., None]}
     * for the upstream gradient {@code upstream}, evaluated with plain loops. Values with {@code valueMask == 0} are
     * excluded from the softmax. Every batch must keep at least one value.
     */
    private static Expected reference(double[][][] q, double[][][] keys, double[][][] values, double[][] queryMask,
                                      double[][] valueMask, double[][][] upstream, double scale) {
        int batch = q.length;
        int queryLength = q[0].length;
        int dim = q[0][0].length;
        int valueLength = keys[0].length;
        int valueDim = values[0][0].length;

        double[][][] out = new double[batch][queryLength][valueDim];
        double[][][] weightsOut = new double[batch][queryLength][valueLength];
        double[][][] dQ = new double[batch][queryLength][dim];
        double[][][] dV = new double[batch][valueLength][valueDim];
        double[][][] dK = new double[batch][valueLength][dim];

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < queryLength; i++) {
                double[] logits = new double[valueLength];
                boolean[] kept = new boolean[valueLength];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < valueLength; j++) {
                    kept[j] = valueMask == null || valueMask[b][j] != 0.0;
                    if (!kept[j]) {
                        continue;
                    }
                    double dot = 0.0;
                    for (int k = 0; k < dim; k++) {
                        dot += q[b][i][k] * keys[b][j][k];
                    }
                    logits[j] = scale * dot;
                    max = Math.max(max, logits[j]);
                }

                double[] weights = new double[valueLength];
                double sum = 0.0;
                for (int j = 0; j < valueLength; j++) {
                    weights[j] = kept[j] ? Math.exp(logits[j] - max) : 0.0;
                    sum += weights[j];
                }
                for (int j = 0; j < valueLength; j++) {
                    weights[j] /= sum;
                }

                weightsOut[b][i] = weights.clone();

                // out = (W V) * queryMask, and the gradient of W V is E' = E * queryMask
                double queryKept = queryMask == null ? 1.0 : queryMask[b][i];
                for (int j = 0; j < valueLength; j++) {
                    for (int d = 0; d < valueDim; d++) {
                        out[b][i][d] += queryKept * weights[j] * values[b][j][d];
                    }
                }

                // dW = E' V^T and dV = W^T E'
                double[] dWeights = new double[valueLength];
                for (int j = 0; j < valueLength; j++) {
                    for (int d = 0; d < valueDim; d++) {
                        double gradient = queryKept * upstream[b][i][d];
                        dWeights[j] += gradient * values[b][j][d];
                        dV[b][j][d] += weights[j] * gradient;
                    }
                }

                // dLogits = W * (dW - rowsum(dW * W)); dQ = scale * dLogits K; dK = scale * dLogits^T Q
                double weighted = 0.0;
                for (int j = 0; j < valueLength; j++) {
                    weighted += weights[j] * dWeights[j];
                }
                for (int j = 0; j < valueLength; j++) {
                    double dLogit = scale * weights[j] * (dWeights[j] - weighted);
                    for (int k = 0; k < dim; k++) {
                        dQ[b][i][k] += dLogit * keys[b][j][k];
                        dK[b][j][k] += dLogit * q[b][i][k];
                    }
                }
            }
        }
        return new Expected(out, weightsOut, dQ, dV, dK);
    }
}
