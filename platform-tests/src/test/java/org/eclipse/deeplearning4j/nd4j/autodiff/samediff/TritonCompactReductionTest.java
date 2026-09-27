package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Large ordered means must compile compactly without changing the native reduction tree. */
class TritonCompactReductionTest {
    @ParameterizedTest
    @ValueSource(ints = {2048, 2051})
    void meanPreservesLaneOrderAndCompilesQuickly(int width) {
        assumeTrue(Nd4j.backends().isCudaAvailable() && Nd4j.getNativeOps().isTritonAvailable());
        boolean previousAll = Nd4j.getEnvironment().tritonCompileAll();
        String previousTypes = Nd4j.getEnvironment().tritonIncludeTypes();
        final int rows = 3072;
        // Exactly representable mixed magnitudes expose reassociation, not just shape errors.
        float[] data = new float[rows * width];
        float[] expected = new float[rows];
        float[] pattern = {65536f, 0.03125f, -65536f, 0.125f};
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < width; k++) {
                data[row * width + k] = pattern[(k + row) % pattern.length];
            }
            float[] lanes = new float[256];
            for (int lane = 0; lane < lanes.length; lane++)
                for (int k = lane; k < width; k += lanes.length)
                    lanes[lane] += data[row * width + k];
            for (int active = 128; active > 0; active >>= 1)
                for (int lane = 0; lane < active; lane++) lanes[lane] += lanes[lane + active];
            expected[row] = lanes[0] / width;
        }
        try (INDArray input = Nd4j.createFromArray(data).reshape(rows, width);
             SameDiff graph = SameDiff.create()) {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            Nd4j.getEnvironment().setTritonIncludeTypes("REDUCTION");
            graph.placeHolder("x", DataType.FLOAT, rows, width).mean("out", 1);
            graph.setGraphExecutionMode(GraphExecutionMode.TRITON);
            for (int iteration = 0; iteration < 6; iteration++) {
                long start = System.nanoTime();
                try (INDArray output = graph.output(Map.of("x", input), "out").get("out")) {
                    long elapsedMs = (System.nanoTime() - start) / 1_000_000;
                    System.out.println("COMPACT_REDUCTION width=" + width + " iteration=" + iteration
                            + " elapsedMs=" + elapsedMs);
                    assertEquals(DataType.FLOAT, output.dataType());
                    float[] actual = output.data().asFloat();
                    assertArrayEquals(expected, actual, 0.0f,
                            "ordered accumulation changed at iteration " + iteration);
                    assertTrue(elapsedMs < 120_000,
                            "one reduction must not spend minutes expanding/compiling IR: " + elapsedMs);
                }
            }
            assertTrue(DspPlanAssertions.getSegmentCompiledBackend(graph, 0).contains("Triton"),
                    "native execution must not conceal a failed Triton compilation");
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(previousAll);
            Nd4j.getEnvironment().setTritonIncludeTypes(previousTypes);
        }
    }
}
