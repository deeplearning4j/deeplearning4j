package org.nd4j.dsp.runtime;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class CpuSlotLastUseTest {
    @Test
    void longChainRecreatesRetiredIntermediatesOnEveryExecution() {
        try (SameDiff graph = SameDiff.create()) {
            graph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            var input = graph.placeHolder("input", DataType.FLOAT, 128, 256);
            var current = input;
            for (int i = 0; i < 12; i++) {
                current = current.add("step_" + i, 1.0);
                current = current.permute(1, 0);
            }
            String outputName = current.name();
            for (int iteration = 0; iteration < 3; iteration++) {
                try (INDArray value = Nd4j.valueArrayOf(new long[]{128, 256}, iteration, DataType.FLOAT)) {
                    Map<String, INDArray> outputs = graph.output(Map.of("input", value), outputName);
                    try {
                        INDArray output = outputs.get(outputName);
                        assertArrayEquals(new long[]{128, 256}, output.shape());
                        assertEquals(iteration + 12.0, output.minNumber().doubleValue(), 0.0);
                        assertEquals(iteration + 12.0, output.maxNumber().doubleValue(), 0.0);
                        assertFalse(value.wasClosed());
                    } finally {
                        outputs.values().forEach(INDArray::close);
                    }
                }
            }
        }
    }

    @Test
    void aliasesAndRequestedIntermediatesSurviveAcrossRepeatedExecutions() {
        try (SameDiff graph = SameDiff.create()) {
            graph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            var input = graph.placeHolder("input", DataType.FLOAT, 2, 3);
            var base = input.add("base", 1.0);
            var view = base.permute(1, 0);
            var branch = base.mul("branch", 2.0);
            var result = view.add("result", branch.permute(1, 0));
            for (int iteration = 0; iteration < 3; iteration++) {
                try (INDArray value = Nd4j.valueArrayOf(new long[]{2, 3}, iteration, DataType.FLOAT)) {
                    Map<String, INDArray> outputs = graph.output(Map.of("input", value), "base", "result");
                    try {
                        assertEquals(iteration + 1.0, outputs.get("base").meanNumber().doubleValue(), 0.0);
                        assertArrayEquals(new long[]{3, 2}, outputs.get("result").shape());
                        assertEquals(3.0 * (iteration + 1), outputs.get("result").meanNumber().doubleValue(), 0.0);
                        assertFalse(value.wasClosed(), "External input remains caller-owned");
                    } finally {
                        outputs.values().forEach(INDArray::close);
                    }
                }
            }
        }
    }
}
