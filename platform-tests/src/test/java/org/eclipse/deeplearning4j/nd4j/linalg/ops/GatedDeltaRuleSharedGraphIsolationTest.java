/*
 * Isolation bit #5: gated_delta_rule through ONE SameDiff graph reused across
 * shape-churn invokes (L=1241 then L=1014) — replicating the model's
 * single-graph-across-calls executor pattern, including DSP freezing between
 * calls. This is the dimension the per-call-graph tests (bits #2/#3) missed.
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

public class GatedDeltaRuleSharedGraphIsolationTest {

    private static final int H = 16;
    private static final int DK = 128;
    private static final int DV = 128;

    private static INDArray hostQkv(int L, long seed) {
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
        return Nd4j.create(fused, new long[]{1, L, 3 * 2048}, 'c');
    }

    private static void fillBetaGate(float[] beta, float[] gate, long seed) {
        Random rng = new Random(seed + 7);
        for (int i = 0; i < beta.length; i++) {
            beta[i] = rng.nextFloat();
            gate[i] = (float) (rng.nextDouble() * -6.0);
        }
    }

    private static long firstNonFinite(INDArray a) {
        float[] f = a.data().asFloat();
        for (int i = 0; i < f.length; i++) {
            if (!Float.isFinite(f[i])) return i;
        }
        return -1;
    }

    @Test
    public void sharedGraphShapeChurnStaysFinite() {
        SameDiff graph = SameDiff.create();
        SDVariable qVar = graph.placeHolder("q", DataType.FLOAT, 'c', -1, -1, H, DK);
        SDVariable kVar = graph.placeHolder("k", DataType.FLOAT, 'c', -1, -1, H, DK);
        SDVariable vVar = graph.placeHolder("v", DataType.FLOAT, 'c', -1, -1, H, DV);
        SDVariable betaVar = graph.placeHolder("beta", DataType.FLOAT, 'c', -1, -1, H);
        SDVariable gateVar = graph.placeHolder("gate", DataType.FLOAT, 'c', -1, -1, H);
        SDVariable stateVar = graph.placeHolder("state", DataType.FLOAT, 'c', 1, H, DK, DV);
        SDVariable actualLengthVar = graph.placeHolder("actual_length", DataType.INT64);
        SDVariable[] graphOutputs = new GatedDeltaRule(
                graph, qVar, kVar, vVar, betaVar, gateVar, stateVar, actualLengthVar)
                .outputVariables();
        graph.updateVariableNameAndReference(graphOutputs[0], "gdr_output");
        graph.updateVariableNameAndReference(graphOutputs[1], "gdr_state");

        for (int round = 0; round < 3; round++) {
            long seed = 100 + round;
            for (int L : new int[]{1241, 1014}) {
                INDArray qkv = hostQkv(L, seed);
                INDArray q = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 2048))
                        .reshape(1, L, H, DK).dup();
                INDArray k = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(2048, 4096))
                        .reshape(1, L, H, DK).dup();
                INDArray v = qkv.get(org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(4096, 6144))
                        .reshape(1, L, H, DV).dup();
                float[] bh = new float[1 * L * H];
                float[] gh = new float[1 * L * H];
                fillBetaGate(bh, gh, seed);
                INDArray beta = Nd4j.create(bh, new long[]{1, L, H}, 'c');
                INDArray gate = Nd4j.create(gh, new long[]{1, L, H}, 'c');
                INDArray state = Nd4j.zeros(DataType.FLOAT, 1, H, DK, DV);
                INDArray actualLen = Nd4j.scalar(DataType.INT64, (long) L);

                Map<String, INDArray> inputs = new LinkedHashMap<>();
                inputs.put("q", q);
                inputs.put("k", k);
                inputs.put("v", v);
                inputs.put("beta", beta);
                inputs.put("gate", gate);
                inputs.put("state", state);
                inputs.put("actual_length", actualLen);
                Map<String, INDArray> out = graph.output(inputs, "gdr_output", "gdr_state");
                long bad = firstNonFinite(out.get("gdr_output"));
                long badState = firstNonFinite(out.get("gdr_state"));
                System.out.printf("GDR_SHARED round=%d L=%d bad=%d badState=%d%n",
                        round, L, bad, badState);
                if (bad >= 0 || badState >= 0) {
                    fail("Shared-graph invoke L=" + L + " round=" + round + " non-finite at "
                            + (bad >= 0 ? bad : badState));
                }
            }
        }
    }
}
