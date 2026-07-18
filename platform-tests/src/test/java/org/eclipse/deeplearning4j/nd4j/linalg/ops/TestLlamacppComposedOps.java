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
 * Regression tests for the P2 llama.cpp-compat composable ops — GLU-family gated
 * activations (swiglu/geglu/reglu), Swin window partition/unpartition
 * (win_part/win_unpart), and the sinusoidal/timestep embedding generators.
 * All are validated against pure-Java double-precision references (correct
 * semantics — several of the original llamacpp impls were nonstandard, e.g.
 * sinusoidal_position_encoding used RoPE).
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestLlamacppComposedOps 2>&1 | tee /tmp/test-llamacpp-composed.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestLlamacppComposedOps {

    private static INDArray exec1(String opName, INDArray in, long[] iArgs) {
        DynamicCustomOp.DynamicCustomOpsBuilder b = DynamicCustomOp.builder(opName).addInputs(in);
        if (iArgs != null && iArgs.length > 0) b.addIntegerArguments(iArgs);
        return Nd4j.exec(b.build())[0];
    }

    private static double silu(double x) { return x / (1.0 + Math.exp(-x)); }
    private static double gelu(double x) {
        // Mirrors nd4j transform::PreciseGELU (ops.h:1071) exactly, which nests the
        // cubic coefficient inside the pow: xp = x + (0.044715*x)^3 (nonstandard vs the
        // textbook x + 0.044715*x^3). geglu is composed from this existing transform.
        double sp = Math.sqrt(2.0 / Math.PI);
        double inner = 0.044715 * x;
        double xp = x + inner * inner * inner;
        return 0.5 * x * (1.0 + Math.tanh(sp * xp));
    }
    private static double relu(double x) { return Math.max(0.0, x); }

    // ── GLU family ───────────────────────────────────────────────────────────

    private void checkGlu(String opName, java.util.function.DoubleUnaryOperator act, double tol) {
        Nd4j.getRandom().setSeed(7);
        int rows = 4, half = 5;
        INDArray x = Nd4j.rand(DataType.FLOAT, rows, 2 * half).subi(0.5).muli(4);
        INDArray out = exec1(opName, x, null);

        assertArrayEquals(new long[]{rows, half}, out.shape(), opName + " output shape");
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < half; j++) {
                double gate = x.getDouble(r, j);
                double up = x.getDouble(r, j + half);
                assertEquals(act.applyAsDouble(gate) * up, out.getDouble(r, j), tol,
                        opName + " mismatch at [" + r + "," + j + "]");
            }
        }
    }

    @Test public void testSwiglu() { checkGlu("swiglu", TestLlamacppComposedOps::silu, 1e-5); }
    @Test public void testGeglu() { checkGlu("geglu", TestLlamacppComposedOps::gelu, 1e-4); }
    @Test public void testReglu() { checkGlu("reglu", TestLlamacppComposedOps::relu, 1e-6); }

    // ── Swin window partition / unpartition ──────────────────────────────────

    @Test
    public void testWinPartUnpartRoundTrip() {
        Nd4j.getRandom().setSeed(11);
        int n = 2, h = 6, w = 4, c = 3, win = 2;
        INDArray x = Nd4j.rand(DataType.FLOAT, n, h, w, c);

        INDArray windows = exec1("win_part", x, new long[]{win});
        assertArrayEquals(new long[]{n * (h / win) * (w / win), win, win, c}, windows.shape());

        INDArray restored = exec1("win_unpart", windows, new long[]{win, h, w});
        assertArrayEquals(x.shape(), restored.shape());
        assertEquals(x, restored, "win_unpart(win_part(x)) must round-trip");
    }

    @Test
    public void testWinPartFirstWindowContent() {
        // window [0] must equal x[0, 0:win, 0:win, :]
        int win = 2;
        INDArray x = Nd4j.arange(0, 1 * 4 * 4 * 2).castTo(DataType.FLOAT).reshape(1, 4, 4, 2);
        INDArray windows = exec1("win_part", x, new long[]{win});
        INDArray w0 = windows.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0));
        INDArray ref = x.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, win),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, win),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        assertEquals(ref, w0, "first window must be the top-left win×win patch");
    }

    // ── Embedding generators ─────────────────────────────────────────────────

    @Test
    public void testTimestepEmbeddingCosSinHalves() {
        int T = 5, dim = 8, maxPeriod = 10000;
        int half = dim / 2;
        INDArray timesteps = Nd4j.createFromArray(0f, 1f, 10f, 100f, 999f);
        INDArray emb = exec1("timestep_embedding", timesteps, new long[]{dim, maxPeriod});

        assertArrayEquals(new long[]{T, dim}, emb.shape());
        for (int t = 0; t < T; t++) {
            double ts = timesteps.getDouble(t);
            for (int j = 0; j < half; j++) {
                double freq = Math.exp(-Math.log(maxPeriod) * j / half);
                double arg = ts * freq;
                assertEquals(Math.cos(arg), emb.getDouble(t, j), 1e-4, "cos half [" + t + "," + j + "]");
                assertEquals(Math.sin(arg), emb.getDouble(t, j + half), 1e-4, "sin half [" + t + "," + j + "]");
            }
        }
    }

    @Test
    public void testSinusoidalPositionEncodingSinCosHalves() {
        int T = 4, dim = 6;
        int half = dim / 2;
        INDArray positions = Nd4j.createFromArray(0f, 1f, 2f, 3f);
        INDArray pe = exec1("sinusoidal_position_encoding", positions, new long[]{dim});

        assertArrayEquals(new long[]{T, dim}, pe.shape());
        for (int t = 0; t < T; t++) {
            double p = positions.getDouble(t);
            for (int j = 0; j < half; j++) {
                double freq = Math.exp(-Math.log(10000.0) * j / half);
                double arg = p * freq;
                assertEquals(Math.sin(arg), pe.getDouble(t, j), 1e-4, "sin half [" + t + "," + j + "]");
                assertEquals(Math.cos(arg), pe.getDouble(t, j + half), 1e-4, "cos half [" + t + "," + j + "]");
            }
        }
    }
}
