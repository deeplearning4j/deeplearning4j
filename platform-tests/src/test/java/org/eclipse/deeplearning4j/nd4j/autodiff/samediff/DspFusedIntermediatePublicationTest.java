package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Requested intermediate outputs must retain their own values through fusion. */
class DspFusedIntermediatePublicationTest {
    @ParameterizedTest
    @EnumSource(value = GraphExecutionMode.class, names = {"AUTO", "CUDA_GRAPHS", "TRITON"})
    void requestedIntermediates(GraphExecutionMode mode) {
        try (INDArray x = Nd4j.ones(DataType.FLOAT, 1, 8);
             INDArray a = Nd4j.ones(DataType.FLOAT, 1, 8);
             INDArray b = Nd4j.valueArrayOf(new long[]{1, 8}, 2.0f);
             INDArray c = Nd4j.valueArrayOf(new long[]{1, 8}, 0.5f);
             SameDiff graph = SameDiff.create()) {
            SDVariable input = graph.placeHolder("x", DataType.FLOAT, 1, 8);
            SDVariable av = graph.placeHolder("a", DataType.FLOAT, 1, 8);
            SDVariable bv = graph.placeHolder("b", DataType.FLOAT, 1, 8);
            SDVariable cv = graph.placeHolder("c", DataType.FLOAT, 1, 8);
            input.add("sum1", av).mul("prod", bv).add("sum2", cv);
            graph.setGraphExecutionMode(mode);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("x", x);
            inputs.put("a", a);
            inputs.put("b", b);
            inputs.put("c", c);
            for (int step = 0; step < 12; step++) {
                x.assign(step + 1);
                Map<String, INDArray> outputs = graph.output(inputs, "sum1", "prod", "sum2");
                double sum = step + 2;
                for (int i = 0; i < 8; i++) {
                    assertEquals(sum, outputs.get("sum1").getDouble(i), 1e-5, mode + " sum1 step=" + step);
                    assertEquals(sum * 2, outputs.get("prod").getDouble(i), 1e-5, mode + " prod step=" + step);
                    assertEquals(sum * 2 + 0.5, outputs.get("sum2").getDouble(i), 1e-5, mode + " sum2 step=" + step);
                }
            }
        }
    }
}
