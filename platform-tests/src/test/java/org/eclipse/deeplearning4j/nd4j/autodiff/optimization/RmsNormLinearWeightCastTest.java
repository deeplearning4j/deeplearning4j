package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

class RmsNormLinearWeightCastTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preservesFloatLogitsWithHalfTransposedWeights(boolean exportCast) {
        try (SameDiff raw = SameDiff.create()) {
            SDVariable x = raw.placeHolder("x", DataType.FLOAT, -1, 32);
            SDVariable gamma = raw.constant("gamma", Nd4j.ones(DataType.FLOAT, 32));
            SDVariable weight = raw.var("weight", Nd4j.valueArrayOf(new long[]{8, 32}, 4096, DataType.HALF));
            SDVariable wide = weight.permute(1, 0).castTo("wide", DataType.FLOAT);
            raw.mmul("logits", raw.nn().rmsNorm(x, gamma, 1e-6), wide);
            String[] outputs = exportCast ? new String[]{"logits", "wide"} : new String[]{"logits"};
            raw.setOutputs(outputs);
            try (SameDiff optimized = GraphOptimizer.optimize(raw, outputs)) {
                var fused = optimized.getOps().values().stream()
                        .filter(op -> "rms_norm_linear".equals(op.getOp().opName())).findFirst();
                assertTrue(fused.isPresent(), "normalization/linear fusion must remain enabled");
                assertEquals(DataType.HALF, optimized.getVariable(fused.get().getInputsToOp().get(2)).dataType());
                assertEquals(DataType.FLOAT, optimized.getVariable("logits").dataType());
                if (exportCast) assertNotNull(optimized.getVariable("wide"), "exported cast must survive");
                else assertFalse(optimized.hasVariable("wide"), "dead expanded weight must be removed");
                for (int rows : new int[]{4, 1, 4}) {
                    try (INDArray input = Nd4j.ones(DataType.FLOAT, rows, 32)) {
                        INDArray actual = optimized.outputSingle(Map.of("x", input), "logits");
                        try {
                            assertArrayEquals(new long[]{rows, 8}, actual.shape());
                            assertEquals(DataType.FLOAT, actual.dataType());
                            // Greater than HALF's finite range: a HALF output or
                            // HALF accumulator cannot accidentally pass this check.
                            float expected = (float)(32.0 * 4096 / Math.sqrt(1.0 + 1e-6));
                            for (float value : actual.data().asFloat()) assertEquals(expected, value, 0.05f);
                        } finally { actual.close(); }
                    }
                }
            }
        }
    }
}
