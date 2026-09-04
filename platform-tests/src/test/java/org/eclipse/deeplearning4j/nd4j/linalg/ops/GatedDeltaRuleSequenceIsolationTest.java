/*
 * ******************************************************************************
 * * Isolation test: gated_delta_rule production-shape sequence on CUDA
 * * Reproduces the model-to-crawl call sequence (L=1241 prefill then L=1014)
 * * with synthetic inputs matched to the captured DSP lineage ranges.
 * ******************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class GatedDeltaRuleSequenceIsolationTest {

    private static final int H = 16;
    private static final int DK = 128;
    private static final int DV = 128;

    private INDArray[] run(int L, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        long[] qkvShape = new long[]{1, L, H, DK};
        long[] vShape = new long[]{1, L, H, DV};

        // Build everything host-side first: no per-element CUDA syncs.
        float[] qh = new float[1 * L * H * DK];
        float[] kh = new float[1 * L * H * DK];
        float[] vh = new float[1 * L * H * DV];
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < H; h++) {
                double qn = 0, kn = 0;
                int base = (t * H + h) * DK;
                float[] qrow = new float[DK];
                float[] krow = new float[DK];
                for (int d = 0; d < DK; d++) {
                    qrow[d] = (float) (rng.nextDouble() * 2 - 1);
                    krow[d] = (float) (rng.nextDouble() * 2 - 1);
                    qn += qrow[d] * qrow[d];
                    kn += krow[d] * krow[d];
                }
                qn = Math.sqrt(qn); kn = Math.sqrt(kn);
                for (int d = 0; d < DK; d++) {
                    qh[base + d] = qrow[d] / (float) qn;
                    kh[base + d] = krow[d] / (float) kn;
                }
                for (int d = 0; d < DV; d++) {
                    vh[(t * H + h) * DV + d] = (float) (rng.nextDouble() * 30 - 15);
                }
            }
        }
        INDArray q = Nd4j.create(qh, qkvShape, 'c');
        INDArray k = Nd4j.create(kh, qkvShape, 'c');
        INDArray v = Nd4j.create(vh, vShape, 'c');

        float[] bh = new float[1 * L * H];
        float[] gh = new float[1 * L * H];
        for (int i = 0; i < bh.length; i++) {
            bh[i] = rng.nextFloat();
            gh[i] = (float) (rng.nextDouble() * -6.0);
        }
        INDArray beta = Nd4j.create(bh, new long[]{1, L, H}, 'c');
        INDArray gate = Nd4j.create(gh, new long[]{1, L, H}, 'c');
        INDArray state = Nd4j.zeros(1, H, DK, DV);
        INDArray actualLen = Nd4j.scalar(org.nd4j.linalg.api.buffer.DataType.INT64, (long) L);

        GatedDeltaRule op = new GatedDeltaRule(q, k, v, beta, gate, state, actualLen);
        INDArray[] out = Nd4j.exec(op);
        return new INDArray[]{out[0], out[1]};
    }

    private static org.nd4j.linalg.api.buffer.DataType DataType_LONG() {
        return org.nd4j.linalg.api.buffer.DataType.INT64;
    }

    private static long firstNonFinite(INDArray a) {
        float[] f = a.data().asFloat();
        for (int i = 0; i < f.length; i++) {
            if (!Float.isFinite(f[i])) return i;
        }
        return -1;
    }

    private void check(String tag, INDArray out, INDArray stateOut) {
        long bad1 = firstNonFinite(out);
        long bad2 = firstNonFinite(stateOut);
        System.out.printf("GDR_ISO %s out.bad=%d stateOut.bad=%d%n", tag, bad1, bad2);
        if (bad1 >= 0 || bad2 >= 0) {
            float[] f = out.data().asFloat();
            int idx = bad1 >= 0 ? (int) bad1 : 0;
            fail(tag + " produced non-finite output at flat index " + (bad1 >= 0 ? bad1 : bad2)
                    + " value=" + (bad1 >= 0 ? f[idx] : Float.NaN));
        }
    }

    @Test
    public void productionSequence1241Then1014StaysFinite() {
        for (int round = 0; round < 3; round++) {
            INDArray[] r1 = run(1241, 42L + round);
            check("round" + round + " L=1241 out", r1[0], r1[1]);

            INDArray[] r2 = run(1014, 42L + round);
            check("round" + round + " L=1014 out", r2[0], r2[1]);
        }
        System.out.println("GDR_ISO sequence complete: " + Arrays.toString(new int[]{3, 2}));
    }
}
