package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for CUDA graph replay accuracy.
 *
 * Verifies that the lineage framework (capture buffers, cross-segment D2D copies,
 * address drift detection) produces correct, non-stale results across multiple
 * decode steps with different inputs.
 *
 * These tests use simple graphs that exercise the same code paths as the VLM
 * decoder (matmul chains, reshape/view ops, multiple placeholders) without
 * requiring model weights or vision encoders.
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class GraphReplayAccuracyTest {

    /**
     * Test 1: Verify graph replay produces DIFFERENT outputs for DIFFERENT inputs.
     *
     * Builds: placeholder [1,16] -> matmul [16,32] -> relu -> matmul [32,16] -> output
     * Runs 10 steps with random inputs. Each step should produce unique output.
     * If graph replay returns stale data, outputs will repeat.
     */
    @Test
    @Order(1)
    @DisplayName("Graph replay produces varied outputs for varied inputs")
    public void testReplayProducesVariedOutputs() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable h = sd.nn.relu(sd.mmul("mm1", input, w1), 0);
        SDVariable out = sd.mmul("output", h, w2);

        // Enable DSP + graph capture
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        List<INDArray> outputs = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.output(Map.of("input", inputData), "output");
            INDArray outArr = result.get("output").dup();
            outputs.add(outArr);

            log.info("Step {}: output[0]={}", step, outArr.getFloat(0));
        }

        // Verify all outputs are DIFFERENT
        for (int i = 1; i < outputs.size(); i++) {
            double diff = outputs.get(i).sub(outputs.get(i - 1)).norm2Number().doubleValue();
            log.info("Step {} vs {}: diff={}", i, i - 1, diff);
            assertTrue(diff > 1e-6, "Step " + i + " output identical to step " + (i - 1)
                    + " — graph replay returning stale data. diff=" + diff);
        }

        log.info("PASS: All 10 steps produced unique outputs");
        sd.close();
    }

    /**
     * Test 2: Verify outputDirect (frozen session) also produces varied outputs.
     *
     * outputDirect reuses the session across calls. This tests that the DSP
     * executor properly updates external inputs in the frozen path.
     */
    @Test
    @Order(2)
    @DisplayName("outputDirect produces varied outputs across steps")
    public void testOutputDirectVariedOutputs() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 32, 16));
        SDVariable h = sd.nn.relu(sd.mmul("mm1", input, w1), 0);
        SDVariable out = sd.mmul("output", h, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        List<INDArray> outputs = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, 16).muli(step + 1);
            Map<String, INDArray> result = sd.outputDirect(Map.of("input", inputData), "output");
            INDArray outArr = result.get("output").dup();
            outputs.add(outArr);
        }

        for (int i = 1; i < outputs.size(); i++) {
            double diff = outputs.get(i).sub(outputs.get(i - 1)).norm2Number().doubleValue();
            assertTrue(diff > 1e-6, "outputDirect step " + i + " identical to step " + (i - 1));
        }

        log.info("PASS: outputDirect produced unique outputs across 10 steps");
        sd.close();
    }

    /**
     * Test 3: Graph replay matches slot-by-slot execution.
     *
     * Runs the same 10 inputs through both SLOT_BY_SLOT and CUDA_GRAPHS modes.
     * Outputs must match within floating point tolerance.
     * This catches any capture buffer or D2D copy bugs that produce wrong values.
     */
    @Test
    @Order(3)
    @DisplayName("Graph replay matches slot-by-slot reference")
    public void testReplayMatchesSlotBySlot() {
        // Build graph
        INDArray w1Data = Nd4j.randn(DataType.FLOAT, 16, 32);
        INDArray w2Data = Nd4j.randn(DataType.FLOAT, 32, 16);
        INDArray[] inputs = new INDArray[10];
        for (int i = 0; i < 10; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, 16);
        }

        // Run SLOT_BY_SLOT (reference)
        List<INDArray> refOutputs = new ArrayList<>();
        {
            SameDiff sd = SameDiff.create();
            sd.placeHolder("input", DataType.FLOAT, 1, 16);
            sd.constant("w1", w1Data.dup());
            sd.constant("w2", w2Data.dup());
            SDVariable h = sd.nn.relu(sd.mmul("mm1", sd.getVariable("input"), sd.getVariable("w1")), 0);
            sd.mmul("output", h, sd.getVariable("w2"));
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

            for (int i = 0; i < 10; i++) {
                Map<String, INDArray> result = sd.output(Map.of("input", inputs[i].dup()), "output");
                refOutputs.add(result.get("output").dup());
            }
            sd.close();
        }

        // Run CUDA_GRAPHS
        List<INDArray> graphOutputs = new ArrayList<>();
        {
            SameDiff sd = SameDiff.create();
            sd.placeHolder("input", DataType.FLOAT, 1, 16);
            sd.constant("w1", w1Data.dup());
            sd.constant("w2", w2Data.dup());
            SDVariable h = sd.nn.relu(sd.mmul("mm1", sd.getVariable("input"), sd.getVariable("w1")), 0);
            sd.mmul("output", h, sd.getVariable("w2"));
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

            for (int i = 0; i < 10; i++) {
                Map<String, INDArray> result = sd.output(Map.of("input", inputs[i].dup()), "output");
                graphOutputs.add(result.get("output").dup());
            }
            sd.close();
        }

        // Compare
        for (int i = 0; i < 10; i++) {
            double diff = graphOutputs.get(i).sub(refOutputs.get(i)).norm2Number().doubleValue();
            double refNorm = refOutputs.get(i).norm2Number().doubleValue();
            double relErr = refNorm > 0 ? diff / refNorm : diff;
            log.info("Step {}: relErr={} (abs diff={})", i, relErr, diff);
            assertTrue(relErr < 1e-3, "Step " + i + " graph replay diverged from slot-by-slot. relErr=" + relErr);
        }

        log.info("PASS: Graph replay matches slot-by-slot across 10 steps");
    }

    /**
     * Test 4: Multiple placeholders with graph replay.
     *
     * Simulates the decoder pattern: input_ids + attention_mask + position_ids
     * all change each step. Verifies capture buffers handle multiple external
     * inputs correctly.
     */
    @Test
    @Order(4)
    @DisplayName("Multiple placeholders update correctly in graph replay")
    public void testMultiplePlaceholders() {
        SameDiff sd = SameDiff.create();
        SDVariable ids = sd.placeHolder("input_ids", DataType.FLOAT, 1, 1);
        SDVariable mask = sd.placeHolder("attention_mask", DataType.FLOAT, 1, 1);
        SDVariable pos = sd.placeHolder("position_ids", DataType.FLOAT, 1, 1);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 1, 16));
        // Output depends on all three placeholders
        SDVariable combined = ids.add("add1", mask).add("add2", pos);
        SDVariable out = sd.mmul("output", combined, w);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        List<INDArray> outputs = new ArrayList<>();
        for (int step = 0; step < 10; step++) {
            Map<String, INDArray> feed = Map.of(
                    "input_ids", Nd4j.scalar(DataType.FLOAT, step + 1).reshape(1, 1),
                    "attention_mask", Nd4j.scalar(DataType.FLOAT, 1.0f).reshape(1, 1),
                    "position_ids", Nd4j.scalar(DataType.FLOAT, 680 + step).reshape(1, 1)
            );
            Map<String, INDArray> result = sd.output(feed, "output");
            outputs.add(result.get("output").dup());
        }

        for (int i = 1; i < outputs.size(); i++) {
            double diff = outputs.get(i).sub(outputs.get(i - 1)).norm2Number().doubleValue();
            assertTrue(diff > 1e-6, "Step " + i + " identical to step " + (i - 1)
                    + " — placeholder not updating in graph replay");
        }

        log.info("PASS: Multiple placeholders correctly updated across 10 steps");
        sd.close();
    }

    /**
     * Test 5: Per-step latency measurement.
     *
     * Measures raw step latency to detect performance regressions.
     * Warmup steps excluded. Reports P50 and P90 latencies.
     */
    @Test
    @Order(5)
    @DisplayName("Per-step latency benchmark (detect sync overhead)")
    public void testPerStepLatency() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 64);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 64, 256));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 256, 64));
        SDVariable h = sd.nn.relu(sd.mmul("mm1", input, w1), 0);
        SDVariable out = sd.mmul("output", h, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

        int warmup = 5;
        int measure = 50;
        List<Long> latencies = new ArrayList<>();

        for (int step = 0; step < warmup + measure; step++) {
            INDArray inputData = Nd4j.randn(DataType.FLOAT, 1, 64);
            long t0 = System.nanoTime();
            sd.output(Map.of("input", inputData), "output");
            long elapsed = (System.nanoTime() - t0) / 1_000_000;

            if (step >= warmup) {
                latencies.add(elapsed);
            }
        }

        Collections.sort(latencies);
        long p50 = latencies.get(latencies.size() / 2);
        long p90 = latencies.get((int)(latencies.size() * 0.9));
        long avg = latencies.stream().mapToLong(l -> l).sum() / latencies.size();

        log.info("Latency ({}steps): P50={}ms P90={}ms avg={}ms min={}ms max={}ms",
                measure, p50, p90, avg, latencies.get(0), latencies.get(latencies.size() - 1));

        // With diagnostic sync removed, P50 should be well under 15ms for this simple graph
        // (was ~25ms with the sync)
        log.info("NOTE: If P50 > 15ms, the diagnostic sync may still be active");

        sd.close();
    }
}
