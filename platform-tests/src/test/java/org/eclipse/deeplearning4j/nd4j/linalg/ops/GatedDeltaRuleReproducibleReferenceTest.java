/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The decode-window gated delta rule is bit-identical to its reproducible definition:
 * per state column, prediction and readout are pairwise-tree dots over D_k in FLOAT32
 * round-to-nearest, and the update is {@code g*s + (beta*(v - g*pred))*k} per row.
 * With a zero log-gate the decay is exactly 1 on every implementation, so Java float
 * arithmetic (IEEE round-to-nearest, no contraction) is an exact oracle. D_k = 128
 * exercises the split-row CUDA path (lanes combine pairwise subtrees); D_k = 96 is not a
 * power of two and exercises the single-lane path. With grouped heads, Q/K carry
 * H_qk heads and value head h reads Q/K head h / (H / H_qk).
 */
class GatedDeltaRuleReproducibleReferenceTest {
    private static final int H = 4;
    private static final int DV = 128;
    private static final int L = 5;

    @ParameterizedTest(name = "dk={0} qkHeads={1}")
    @CsvSource({"128,4", "96,4", "128,2", "96,2", "128,1"})
    void decodeWindowMatchesReferenceBitwise(int dk, int qkHeads) {
        final int group = H / qkHeads;
        Random rng = new Random(20260928L + dk * 31L + qkHeads);
        float[] q = random(rng, L * qkHeads * dk, 1f);
        float[] k = random(rng, L * qkHeads * dk, 0.2f);
        float[] v = random(rng, L * H * DV, 4f);
        float[] beta = new float[L * H];
        for (int i = 0; i < beta.length; i++) beta[i] = rng.nextFloat();
        float[] state = random(rng, H * dk * DV, 1f);

        GatedDeltaRule op = new GatedDeltaRule(
                Nd4j.create(q, new long[]{1, L, qkHeads, dk}, 'c'),
                Nd4j.create(k, new long[]{1, L, qkHeads, dk}, 'c'),
                Nd4j.create(v, new long[]{1, L, H, DV}, 'c'),
                Nd4j.create(beta, new long[]{1, L, H}, 'c'),
                Nd4j.zeros(DataType.FLOAT, 1, L, H),
                Nd4j.create(state.clone(), new long[]{1, H, dk, DV}, 'c'),
                Nd4j.scalar(DataType.INT64, (long) L));
        INDArray[] result = Nd4j.exec(op);
        float[] out = result[0].dup('c').data().asFloat();
        float[] stateOut = result[1].dup('c').data().asFloat();

        float[] column = new float[dk];
        for (int h = 0; h < H; h++) {
            for (int dv = 0; dv < DV; dv++) {
                for (int d = 0; d < dk; d++) column[d] = state[(h * dk + d) * DV + dv];
                for (int t = 0; t < L; t++) {
                    int kqBase = (t * qkHeads + h / group) * dk;
                    float prediction = pairwiseDot(column, k, kqBase, dk);
                    float betaDelta = beta[t * H + h] * (v[(t * H + h) * DV + dv] - prediction);
                    for (int d = 0; d < dk; d++) column[d] = column[d] + betaDelta * k[kqBase + d];
                    float expected = pairwiseDot(column, q, kqBase, dk);
                    int outIndex = (t * H + h) * DV + dv;
                    assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(out[outIndex]),
                            "out t=" + t + " h=" + h + " dv=" + dv + " expected " + expected + " got " + out[outIndex]);
                }
                for (int d = 0; d < dk; d++) {
                    float got = stateOut[(h * dk + d) * DV + dv];
                    assertEquals(Float.floatToRawIntBits(column[d]), Float.floatToRawIntBits(got),
                            "state h=" + h + " dk=" + d + " dv=" + dv);
                }
            }
        }
    }

    /**
     * Grouped Q/K heads are the same recurrence as Q/K repeated across each group, on
     * every path, including the chunked prefill (L >= 64, no actualLen): outputs and
     * states must be bit-identical.
     */
    @ParameterizedTest(name = "L={0}")
    @CsvSource({"128", "5"})
    void groupedHeadsEqualRepeatedHeads(int length) {
        final int dk = 128, qkHeads = 2, group = H / qkHeads;
        Random rng = new Random(20260929L + length);
        float[] q = random(rng, length * qkHeads * dk, 1f);
        float[] k = random(rng, length * qkHeads * dk, 0.2f);
        float[] v = random(rng, length * H * DV, 4f);
        float[] beta = new float[length * H];
        float[] gate = new float[length * H];
        for (int i = 0; i < beta.length; i++) {
            beta[i] = rng.nextFloat();
            gate[i] = -rng.nextFloat();
        }
        float[] state = random(rng, H * dk * DV, 1f);
        float[] qRepeated = new float[length * H * dk];
        float[] kRepeated = new float[length * H * dk];
        for (int t = 0; t < length; t++) {
            for (int h = 0; h < H; h++) {
                System.arraycopy(q, (t * qkHeads + h / group) * dk, qRepeated, (t * H + h) * dk, dk);
                System.arraycopy(k, (t * qkHeads + h / group) * dk, kRepeated, (t * H + h) * dk, dk);
            }
        }
        INDArray[] grouped = run(q, k, qkHeads, v, beta, gate, state, length, dk);
        INDArray[] repeated = run(qRepeated, kRepeated, H, v, beta, gate, state, length, dk);
        for (int i = 0; i < 2; i++) {
            float[] a = grouped[i].dup('c').data().asFloat();
            float[] b = repeated[i].dup('c').data().asFloat();
            assertEquals(b.length, a.length);
            for (int j = 0; j < a.length; j++) {
                assertEquals(Float.floatToRawIntBits(b[j]), Float.floatToRawIntBits(a[j]),
                        (i == 0 ? "output" : "state") + " element " + j + " L=" + length);
            }
        }
    }

    private static INDArray[] run(float[] q, float[] k, int qkHeads, float[] v, float[] beta, float[] gate,
                                  float[] state, int length, int dk) {
        GatedDeltaRule op = new GatedDeltaRule(
                Nd4j.create(q, new long[]{1, length, qkHeads, dk}, 'c'),
                Nd4j.create(k, new long[]{1, length, qkHeads, dk}, 'c'),
                Nd4j.create(v, new long[]{1, length, H, DV}, 'c'),
                Nd4j.create(beta, new long[]{1, length, H}, 'c'),
                Nd4j.create(gate, new long[]{1, length, H}, 'c'),
                Nd4j.create(state.clone(), new long[]{1, H, dk, DV}, 'c'));
        return Nd4j.exec(op);
    }

    /** Pairwise tree in the same order as the native reproducible dot. */
    private static float pairwiseDot(float[] column, float[] vector, int offset, int length) {
        float[] levels = new float[32];
        for (int index = 0; index < length; index++) {
            float value = column[index] * vector[offset + index];
            int completed = index + 1;
            int level = 0;
            while ((completed & 1) == 0) {
                value = levels[level] + value;
                completed >>= 1;
                level++;
            }
            levels[level] = value;
        }
        float result = 0f;
        boolean initialized = false;
        for (int level = 0; level < 32; level++) {
            if ((length & (1 << level)) == 0) continue;
            result = initialized ? levels[level] + result : levels[level];
            initialized = true;
        }
        return result;
    }

    private static float[] random(Random rng, int n, float scale) {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = (rng.nextFloat() * 2f - 1f) * scale;
        return values;
    }
}
