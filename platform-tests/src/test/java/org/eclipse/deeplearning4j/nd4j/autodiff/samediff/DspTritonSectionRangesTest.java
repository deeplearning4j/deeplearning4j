/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Sectioned Triton kernels give independent sections of one launch phase
 * disjoint program ranges so they run concurrently. Sections joined by
 * element-aligned flow (a consumer reading its producer's output at the same
 * indices) must keep a shared range: their order is guaranteed only because
 * the same program runs both. This graph has both kinds — the MTP draft
 * head's pattern of an ordered SERIAL_FMA matmul whose output feeds an
 * elementwise epilogue, next to an independent matmul on the same input —
 * and checks every warmup, capture and replay step against an exact
 * reference. A shared-range violation races and returns stale values.
 */
class DspTritonSectionRangesTest {
    private static final int ROWS = 5;
    private static final int DEPTH = 64;
    private static final int COLUMNS = 48;

    @Test
    void alignedSectionFlowSurvivesConcurrentSectionRanges() {
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Requires CUDA with Triton");
        var environment = Nd4j.getEnvironment();
        boolean enabled = InferenceSession.isDynamicShapePlanEnabled();
        boolean capture = environment.tritonGraphCapture();
        boolean compileAll = environment.tritonCompileAll();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            environment.setTritonGraphCapture(true);
            environment.setTritonCompileAll(true);
            try (SameDiff sd = SameDiff.create();
                 INDArray input = Nd4j.create(DataType.FLOAT, ROWS, DEPTH);
                 INDArray residual = Nd4j.create(DataType.FLOAT, ROWS, COLUMNS);
                 INDArray gatedWeights = Nd4j.create(DataType.FLOAT, DEPTH, COLUMNS);
                 INDArray plainWeights = Nd4j.create(DataType.FLOAT, DEPTH, COLUMNS)) {
                fill(gatedWeights, 1);
                fill(plainWeights, 2);
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, ROWS, DEPTH);
                SDVariable r = sd.placeHolder("r", DataType.FLOAT, ROWS, COLUMNS);
                SDVariable wGated = sd.constant("wGated", gatedWeights);
                SDVariable wPlain = sd.constant("wPlain", plainWeights);
                SDVariable gated = sd.nn().sigmoid(x).mul(x);
                SDVariable projected = serial(sd, gated, wGated);
                projected.add("gatedOut", r);
                serial(sd, x, wPlain).mul("plainOut", 2.0);

                for (int step = 0; step < 32; step++) {
                    fill(input, 10 + step);
                    fill(residual, 20 + step);
                    Map<String, INDArray> outputs = sd.output(
                            Map.of("x", input, "r", residual), "gatedOut", "plainOut");
                    assertMatches(expectedGated(input, gatedWeights, residual), outputs.get("gatedOut"),
                            "gatedOut step=" + step);
                    assertMatches(expectedPlain(input, plainWeights), outputs.get("plainOut"),
                            "plainOut step=" + step);
                }
                DspPlanAssertions.assertTotalGraphReplaysAtLeast(sd, 1, "section ranges must reach replay");
            }
        } finally {
            environment.setTritonCompileAll(compileAll);
            environment.setTritonGraphCapture(capture);
            InferenceSession.setDynamicShapePlanEnabled(enabled);
        }
    }

    private static SDVariable serial(SameDiff sd, SDVariable a, SDVariable b) {
        return new Mmul(sd, a, b, MMulTranspose.allFalse(), Mmul.Arithmetic.SERIAL_FMA, DataType.FLOAT)
                .outputVariable();
    }

    /** Deterministic values in [-1, 1) that change with the seed. */
    private static void fill(INDArray array, int seed) {
        long length = array.length();
        for (long i = 0; i < length; i++) {
            array.putScalar(i, ((i * 37 + seed * 11) % 97) / 48.5 - 1.0);
        }
    }

    // SERIAL_FMA: one FP32 accumulator per output, fma in ascending K from +0.
    private static float serialDot(float[] a, INDArray w, int column) {
        float sum = 0.0f;
        for (int k = 0; k < DEPTH; k++) sum = Math.fma(a[k], w.getFloat(k, column), sum);
        return sum;
    }

    private static float[][] expectedGated(INDArray x, INDArray w, INDArray r) {
        float[][] expected = new float[ROWS][COLUMNS];
        for (int row = 0; row < ROWS; row++) {
            float[] gated = new float[DEPTH];
            for (int k = 0; k < DEPTH; k++) {
                float v = x.getFloat(row, k);
                gated[k] = (float) (1.0 / (1.0 + Math.exp(-v))) * v;
            }
            for (int col = 0; col < COLUMNS; col++)
                expected[row][col] = serialDot(gated, w, col) + r.getFloat(row, col);
        }
        return expected;
    }

    private static float[][] expectedPlain(INDArray x, INDArray w) {
        float[][] expected = new float[ROWS][COLUMNS];
        for (int row = 0; row < ROWS; row++) {
            float[] a = new float[DEPTH];
            for (int k = 0; k < DEPTH; k++) a[k] = x.getFloat(row, k);
            for (int col = 0; col < COLUMNS; col++) expected[row][col] = serialDot(a, w, col) * 2.0f;
        }
        return expected;
    }

    // The plain path is exact by the SERIAL_FMA contract; the gated path carries
    // the sigmoid's last-ulp freedom, far below any stale-value race.
    private static void assertMatches(float[][] expected, INDArray actual, String what) {
        for (int row = 0; row < ROWS; row++)
            for (int col = 0; col < COLUMNS; col++)
                assertEquals(expected[row][col], actual.getFloat(row, col),
                        1e-4f * Math.max(1.0f, Math.abs(expected[row][col])),
                        what + " row=" + row + " col=" + col);
    }
}
