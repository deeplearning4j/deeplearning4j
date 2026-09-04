/*
 * Isolation bit #3: gated_delta_rule with the model's HALF->FLOAT input cast
 * chain. The Qwen graph feeds HALF strided-slices of a fused qkv projection
 * into FLOAT casts, HALF beta/gate into FLOAT casts, and a FLOAT zero state.
 * This is the exact dtype topology from the captured DSP lineage.
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.fail;

public class GatedDeltaRuleHalfCastIsolationTest {

    private static final int H = 16;
    private static final int DK = 128;
    private static final int DV = 128;

    private static INDArray hostQkvHalf(int L, long seed) {
        Random rng = new java.util.Random(seed);
        float[] fused = new float[1 * L * 3 * 2048];
        for (int t = 0; t < L; t++) {
            for (int h = 0; h < H; h++) {
                double qn = 0, kn = 0;
                int qBase = t * 3 * 2048 + h * DK;
                int kBase = t * 3 * 2048 + 2048 + h * DK;
                int vBase = t * 3 * 2048 + 2 * 2048 + h * DV;
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
                    fused[qBase + d] = qrow[d] / (float) qn;
                    fused[kBase + d] = krow[d] / (float) kn;
                }
                for (int d = 0; d < DV; d++) {
                    fused[vBase + d] = (float) (rng.nextDouble() * 30 - 15);
                }
            }
        }
        return Nd4j.create(fused, new long[]{1, L, 3 * 2048}, 'c').castTo(DataType.HALF);
    }

    private static long firstNonFinite(INDArray a) {
        float[] f = a.data().asFloat();
        for (int i = 0; i < f.length; i++) {
            if (!Float.isFinite(f[i])) return i;
        }
        return -1;
    }

    @Test
    public void halfCastInvokeSequenceStaysFinite() {
        for (int round = 0; round < 3; round++) {
            long seed = 200 + round;
            for (int L : new int[]{1241, 1014}) {
                INDArray qkv = hostQkvHalf(L, seed);
                INDArray qHalf = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 2048))
                        .reshape(1, L, H, DK).dup();
                INDArray kHalf = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(2048, 4096))
                        .reshape(1, L, H, DK).dup();
                INDArray vHalf = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(4096, 6144))
                        .reshape(1, L, H, DV).dup();
                Random rng = new Random(seed + 7);
                float[] bh = new float[1 * L * H];
                float[] gh = new float[1 * L * H];
                for (int i = 0; i < bh.length; i++) {
                    bh[i] = rng.nextFloat();
                    gh[i] = (float) (rng.nextDouble() * -6.0);
                }
                INDArray betaHalf = Nd4j.create(bh, new long[]{1, L, H}, 'c').castTo(DataType.HALF);
                INDArray gateHalf = Nd4j.create(gh, new long[]{1, L, H}, 'c').castTo(DataType.HALF);
                INDArray state = Nd4j.zeros(DataType.FLOAT, 1, H, DK, DV);
                INDArray actualLen = Nd4j.scalar(DataType.INT64, (long) L);

                SameDiff graph = SameDiff.create();
                SDVariable qkvVar = graph.placeHolder("qkv", DataType.HALF, qkv.shape());
                SDVariable betaVar = graph.placeHolder("beta", DataType.HALF, betaHalf.shape());
                SDVariable gateVar = graph.placeHolder("gate", DataType.HALF, gateHalf.shape());
                SDVariable stateVar = graph.placeHolder("state", DataType.FLOAT, state.shape());
                SDVariable actualLengthVar = graph.placeHolder("actual_length", DataType.INT64);

                SDVariable qF = qkvVar
                        .get(org.nd4j.autodiff.samediff.SDIndex.point(0L),
                                org.nd4j.autodiff.samediff.SDIndex.all(),
                                org.nd4j.autodiff.samediff.SDIndex.interval((long) 0, (long) 2048))
                        .castTo(DataType.FLOAT).reshape(1, -1, H, DK);
                SDVariable kF = qkvVar
                        .get(org.nd4j.autodiff.samediff.SDIndex.point(0L),
                                org.nd4j.autodiff.samediff.SDIndex.all(),
                                org.nd4j.autodiff.samediff.SDIndex.interval((long) 2048, (long) 4096))
                        .castTo(DataType.FLOAT).reshape(1, -1, H, DK);
                SDVariable vF = qkvVar
                        .get(org.nd4j.autodiff.samediff.SDIndex.point(0L),
                                org.nd4j.autodiff.samediff.SDIndex.all(),
                                org.nd4j.autodiff.samediff.SDIndex.interval((long) 4096, (long) 6144))
                        .castTo(DataType.FLOAT).reshape(1, -1, H, DV);

                SDVariable[] graphOutputs = new GatedDeltaRule(graph,
                        qF, kF, vF,
                        betaVar.castTo(DataType.FLOAT),
                        gateVar.castTo(DataType.FLOAT),
                        stateVar, actualLengthVar).outputVariables();
                graph.updateVariableNameAndReference(graphOutputs[0], "gdr_output");

                Map<String, INDArray> inputs = new LinkedHashMap<>();
                inputs.put("qkv", qkv);
                inputs.put("beta", betaHalf);
                inputs.put("gate", gateHalf);
                inputs.put("state", state);
                inputs.put("actual_length", actualLen);
                Map<String, INDArray> out = graph.output(inputs, "gdr_output");
                long bad = firstNonFinite(out.get("gdr_output"));
                System.out.printf("GDR_HALF round=%d L=%d bad=%d%n", round, L, bad);
                if (bad >= 0) {
                    float[] f = out.get("gdr_output").data().asFloat();
                    fail("HALF-cast invoke L=" + L + " round=" + round
                            + " non-finite at " + bad + " value=" + f[(int) bad]);
                }
            }
        }
    }
}
