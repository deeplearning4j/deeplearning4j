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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Individual-op tests for the native {@code token_sample} op (temperature / top-k / top-p / greedy).
 *
 * <p>Backend-agnostic correctness properties — the same assertions must hold on CPU and CUDA. Run on
 * CUDA these verify the top-k / top-p fix (previously the CUDA kernel ignored top-k / top-p, so
 * {@code topK=1} would <em>not</em> reliably return the argmax). Run on CPU they validate the
 * expectations themselves.</p>
 *
 * Run: {@code mvn test -Dtest=TokenSampleParityTest -Dbackend.artifactId=nd4j-cuda-12.9} (and
 * {@code -Dbackend.artifactId=nd4j-native} for the CPU sanity pass).
 */
class TokenSampleParityTest {

    private static final int V = 48;

    /** Distinct logits with a clear argmax at index 17. */
    private static float[] logitsData() {
        float[] d = new float[V];
        for (int i = 0; i < V; i++) d[i] = (float) (Math.sin(i * 1.37) * 2.0);
        d[17] = 10.0f;   // unambiguous argmax
        return d;
    }

    private static int argmax(float[] d) {
        int best = 0;
        for (int i = 1; i < d.length; i++) if (d[i] > d[best]) best = i;
        return best;
    }

    /** Indices of the n largest logits. */
    private static Set<Integer> topNSet(float[] d, int n) {
        Integer[] idx = new Integer[d.length];
        for (int i = 0; i < d.length; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Float.compare(d[b], d[a]));
        Set<Integer> s = new HashSet<>();
        for (int i = 0; i < n && i < idx.length; i++) s.add(idx[i]);
        return s;
    }

    private static long sample(float[] d, double temp, int topK, double topP, long seed) {
        INDArray logits = Nd4j.create(d, new long[]{1, d.length});
        TokenSample op = new TokenSample(logits, temp, topK, topP, seed);
        INDArray[] out = Nd4j.getExecutioner().exec(op);
        return out[0].getLong(0);
    }

    @Test
    void greedyReturnsArgmax() {
        float[] d = logitsData();
        int am = argmax(d);
        // greedy via temperature=0
        assertEquals(am, sample(d, 0.0, 0, 0.0, 0), "temp=0 must be argmax");
        // greedy via the no-arg-sampling ctor
        INDArray logits = Nd4j.create(d, new long[]{1, V});
        INDArray[] out = Nd4j.getExecutioner().exec(new TokenSample(logits));
        assertEquals(am, out[0].getLong(0), "greedy ctor must be argmax");
    }

    @Test
    void topK1AlwaysArgmax() {
        float[] d = logitsData();
        int am = argmax(d);
        for (long seed = 1; seed <= 64; seed++) {
            assertEquals(am, sample(d, 1.0, 1, 0.0, seed),
                    "topK=1 must collapse to argmax regardless of seed (seed=" + seed + ")");
        }
    }

    @Test
    void tinyTopPAlwaysArgmax() {
        float[] d = logitsData();
        int am = argmax(d);
        for (long seed = 1; seed <= 64; seed++) {
            assertEquals(am, sample(d, 1.0, 0, 0.001, seed),
                    "top-p≈0 nucleus is just the argmax (seed=" + seed + ")");
        }
    }

    @Test
    void topKRestrictsToTopKSet() {
        float[] d = logitsData();
        Set<Integer> top5 = topNSet(d, 5);
        for (long seed = 1; seed <= 200; seed++) {
            int tok = (int) sample(d, 2.0, 5, 0.0, seed);   // high temp = spread mass, still must stay in top-5
            assertTrue(top5.contains(tok),
                    "topK=5 sample " + tok + " must be within the top-5 set " + top5 + " (seed=" + seed + ")");
        }
    }

    @Test
    void topKTopPCombinedStaysInTopK() {
        float[] d = logitsData();
        Set<Integer> top10 = topNSet(d, 10);
        for (long seed = 1; seed <= 200; seed++) {
            int tok = (int) sample(d, 1.0, 10, 0.9, seed);
            assertTrue(top10.contains(tok),
                    "topK=10,topP=0.9 sample " + tok + " must be within the top-10 set (seed=" + seed + ")");
        }
    }

    @Test
    void batchTopK1EachRowArgmax() {
        int B = 3;
        float[][] rows = new float[B][];
        int[] am = new int[B];
        float[] flat = new float[B * V];
        for (int b = 0; b < B; b++) {
            float[] d = new float[V];
            for (int i = 0; i < V; i++) d[i] = (float) (Math.cos(i * 0.7 + b) * 2.0);
            d[(b * 7 + 3) % V] = 9.0f;         // distinct argmax per row
            rows[b] = d;
            am[b] = argmax(d);
            System.arraycopy(d, 0, flat, b * V, V);
        }
        INDArray logits = Nd4j.create(flat, new long[]{B, V});
        INDArray[] out = Nd4j.getExecutioner().exec(new TokenSample(logits, 1.0, 1, 0.0, 123));
        for (int b = 0; b < B; b++) {
            assertEquals(am[b], out[0].getLong(b), "batch row " + b + " topK=1 must be its argmax");
        }
    }

    /** CPU greedy contract: start at (-inf, 0), replace only on a strictly larger value. */
    private static int cpuArgmax(float[] d, int from, int length) {
        float best = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int v = 0; v < length; v++) {
            if (d[from + v] > best) {
                best = d[from + v];
                idx = v;
            }
        }
        return idx;
    }

    /**
     * Greedy selection at decode vocabulary sizes, including the unit-stride vector scan's
     * tail, rank-3 last-position rows (whose row offset may break 16-byte alignment), exact
     * ties (lowest index wins), rows with no value above -inf (index 0) and NaN entries
     * (never selected) — all against the CPU argmax contract.
     */
    @ParameterizedTest(name = "greedy vocab={0} batch={1} seq={2}")
    @CsvSource({"48,1,1", "1023,2,1", "248320,1,1", "248321,3,1", "248320,2,3", "4099,2,2"})
    void greedyMatchesCpuContractAtDecodeSizes(int vocab, int batch, int seq) {
        java.util.Random random = new java.util.Random(vocab * 31L + batch * 7L + seq);
        float[] flat = new float[batch * seq * vocab];
        for (int i = 0; i < flat.length; i++) flat[i] = (float) random.nextGaussian();
        for (int b = 0; b < batch; b++) {
            int row = (b * seq + seq - 1) * vocab;   // the position greedy selection reads
            switch (b % 3) {
                case 0: {   // exact tie at three indices: the lowest wins
                    int[] at = {vocab / 3, vocab - 1, vocab / 7};
                    for (int a : at) flat[row + a] = 50.0f;
                    flat[row] = Float.NaN;   // leading NaN must not stick
                    break;
                }
                case 1: {   // nothing above -inf, NaN mixed in: CPU returns 0
                    for (int v = 0; v < vocab; v++) flat[row + v] = Float.NEGATIVE_INFINITY;
                    flat[row + vocab / 2] = Float.NaN;
                    break;
                }
                default: {  // NaN just after the true maximum
                    int at = (vocab * 5) / 8;
                    flat[row + at] = 40.0f;
                    flat[row + Math.min(at + 1, vocab - 1)] = Float.NaN;
                }
            }
        }
        long[] shape = seq == 1 ? new long[]{batch, vocab} : new long[]{batch, seq, vocab};
        INDArray logits = Nd4j.create(flat, shape);
        INDArray[] out = Nd4j.getExecutioner().exec(new TokenSample(logits));
        for (int b = 0; b < batch; b++) {
            int row = (b * seq + seq - 1) * vocab;
            assertEquals(cpuArgmax(flat, row, vocab), out[0].getLong(b),
                    "vocab=" + vocab + " batch row " + b + " seq=" + seq);
        }
    }
}
