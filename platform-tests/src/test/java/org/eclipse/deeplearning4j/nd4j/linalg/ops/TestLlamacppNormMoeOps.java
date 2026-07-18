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
 * Regression tests for the P3 llama.cpp-compat native ops: group_norm,
 * l2_normalize, load_balance_loss (Switch-Transformer aux loss — corrects the
 * nonstandard llamacpp reduction), sparse_mul_mat (per-token expert matmul),
 * and kv_cache_attention (non-causal SDPA adapter). Validated against
 * pure-Java double-precision references.
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestLlamacppNormMoeOps 2>&1 | tee /tmp/test-llamacpp-normmoe.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestLlamacppNormMoeOps {

    private static INDArray exec(String opName, INDArray[] in, double[] t, long[] i) {
        DynamicCustomOp.DynamicCustomOpsBuilder b = DynamicCustomOp.builder(opName).addInputs(in);
        if (t != null && t.length > 0) {
            Double[] boxed = new Double[t.length];
            for (int k = 0; k < t.length; k++) boxed[k] = t[k];
            b.addFloatingPointArguments(boxed);
        }
        if (i != null && i.length > 0) b.addIntegerArguments(i);
        return Nd4j.exec(b.build())[0];
    }

    @Test
    public void testGroupNorm() {
        Nd4j.getRandom().setSeed(3);
        int N = 2, C = 6, H = 2, W = 2, G = 3;
        double eps = 1e-5;
        INDArray x = Nd4j.rand(DataType.FLOAT, N, C, H, W).muli(3).subi(1.5);
        INDArray out = exec("group_norm", new INDArray[]{x}, new double[]{eps}, new long[]{G});
        assertArrayEquals(x.shape(), out.shape());

        int cpg = C / G, groupSize = cpg * H * W;
        for (int n = 0; n < N; n++) {
            for (int g = 0; g < G; g++) {
                // population mean/var over the group's (C/G, H, W) elements
                double sum = 0;
                double[] vals = new double[groupSize];
                int idx = 0;
                for (int cc = 0; cc < cpg; cc++)
                    for (int hh = 0; hh < H; hh++)
                        for (int ww = 0; ww < W; ww++) {
                            double v = x.getDouble(n, g * cpg + cc, hh, ww);
                            vals[idx++] = v; sum += v;
                        }
                double mean = sum / groupSize, var = 0;
                for (double v : vals) var += (v - mean) * (v - mean);
                var /= groupSize;
                double inv = 1.0 / Math.sqrt(var + eps);
                idx = 0;
                for (int cc = 0; cc < cpg; cc++)
                    for (int hh = 0; hh < H; hh++)
                        for (int ww = 0; ww < W; ww++)
                            assertEquals((vals[idx++] - mean) * inv,
                                    out.getDouble(n, g * cpg + cc, hh, ww), 1e-4,
                                    "group_norm [" + n + "," + g + "]");
            }
        }
    }

    @Test
    public void testL2Normalize() {
        Nd4j.getRandom().setSeed(4);
        int rows = 3, cols = 5;
        double eps = 1e-12;
        INDArray x = Nd4j.rand(DataType.FLOAT, rows, cols).muli(2).subi(1);
        INDArray out = exec("l2_normalize", new INDArray[]{x}, new double[]{eps}, null);
        assertArrayEquals(x.shape(), out.shape());
        for (int r = 0; r < rows; r++) {
            double ss = 0;
            for (int c = 0; c < cols; c++) ss += x.getDouble(r, c) * x.getDouble(r, c);
            double inv = 1.0 / Math.sqrt(ss + eps);
            for (int c = 0; c < cols; c++)
                assertEquals(x.getDouble(r, c) * inv, out.getDouble(r, c), 1e-5, "l2 [" + r + "," + c + "]");
        }
    }

    @Test
    public void testLoadBalanceLoss() {
        int B = 4, E = 3;
        INDArray probs = Nd4j.rand(DataType.FLOAT, B, E);
        // one-hot argmax mask
        INDArray mask = Nd4j.zeros(DataType.FLOAT, B, E);
        for (int b = 0; b < B; b++) mask.putScalar(b, probs.getRow(b).argMax().getInt(0), 1.0);

        INDArray out = exec("load_balance_loss", new INDArray[]{probs, mask}, null, null);
        assertEquals(1, out.length());

        double loss = 0;
        for (int e = 0; e < E; e++) {
            double mp = 0, mm = 0;
            for (int b = 0; b < B; b++) { mp += probs.getDouble(b, e); mm += mask.getDouble(b, e); }
            loss += (mp / B) * (mm / B);
        }
        loss *= E;
        assertEquals(loss, out.getDouble(0), 1e-5);
    }

    @Test
    public void testSparseMulMat() {
        Nd4j.getRandom().setSeed(5);
        int T = 4, H = 3, D = 6, expertsE = 3;
        INDArray input = Nd4j.rand(DataType.FLOAT, T, H).subi(0.5);
        INDArray weights = Nd4j.rand(DataType.FLOAT, expertsE, H, D).subi(0.5);
        INDArray indices = Nd4j.createFromArray(2L, 0L, 1L, 2L);

        INDArray out = exec("sparse_mul_mat", new INDArray[]{input, weights, indices}, null, null);
        assertArrayEquals(new long[]{T, D}, out.shape());
        for (int t = 0; t < T; t++) {
            long e = indices.getLong(t);
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int h = 0; h < H; h++) acc += input.getDouble(t, h) * weights.getDouble(e, h, d);
                assertEquals(acc, out.getDouble(t, d), 1e-4, "sparse_mul_mat [" + t + "," + d + "]");
            }
        }
    }

    @Test
    public void testKvCacheAttentionMatchesGqa() {
        Nd4j.getRandom().setSeed(6);
        INDArray q = Nd4j.rand(DataType.FLOAT, 1, 3, 2, 8);
        INDArray k = Nd4j.rand(DataType.FLOAT, 1, 5, 2, 8);
        INDArray v = Nd4j.rand(DataType.FLOAT, 1, 5, 2, 8);

        INDArray viaCompat = exec("kv_cache_attention", new INDArray[]{q, k, v}, new double[]{0.0}, null);

        // grouped_query_attention, non-causal, auto scale
        DynamicCustomOp gqa = DynamicCustomOp.builder("grouped_query_attention")
                .addInputs(q, k, v).addFloatingPointArguments(0.0).addBooleanArguments(false).build();
        INDArray viaGqa = Nd4j.exec(gqa)[0];

        assertArrayEquals(q.shape(), viaCompat.shape());
        assertEquals(viaGqa, viaCompat);
    }
}
