package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests memory with a large model (30 transformer layers ≈ SmolDocling scale).
 * Each layer: matmul(hidden→4*hidden) → relu → matmul(4*hidden→hidden) → add(residual)
 * Total: ~120 ops, ~2.5GB weights.
 *
 * Run: cd platform-tests && mvn test -Dtest=TestDSPLargeModelMemory \
 *   -Dbackend.artifactId=nd4j-cuda-12.9 -Dlibnd4j.triton=ON 2>&1 | tee /tmp/large-model-mem.log
 */
@Slf4j
public class TestDSPLargeModelMemory {

    @Test
    public void testLargeModelMemoryGrowth() {
        int hidden = 576;
        int inter = 1536;
        int numLayers = 30;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, hidden);
        SDVariable current = sd.reshape(x, 1, hidden);

        // Build 30 transformer-like layers
        for (int layer = 0; layer < numLayers; layer++) {
            String prefix = "l" + layer;
            SDVariable wUp = sd.constant(prefix + "_up", Nd4j.randn(DataType.FLOAT, hidden, inter).muli(0.01));
            SDVariable wDown = sd.constant(prefix + "_down", Nd4j.randn(DataType.FLOAT, inter, hidden).muli(0.01));

            SDVariable h = sd.mmul(prefix + "_h", current, wUp);
            SDVariable act = sd.nn.relu(prefix + "_act", h, 0);
            SDVariable down = sd.mmul(prefix + "_down_mm", act, wDown);
            current = down.add(prefix + "_res", current);
        }

        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, hidden, 100).muli(0.01));
        SDVariable out = sd.mmul("out", current, wOut);

        log.info("Model: {} layers, {} ops", numLayers, sd.ops().length);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, hidden);

        // Warmup
        for (int i = 0; i < 5; i++) {
            sd.outputDirect(Map.of("x", input), "out");
        }

        Nd4j.getExecutioner().commit();
        long freeBefore = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        log.info("Free before 100 steps: {} MB (total={} MB)",
                freeBefore / (1024 * 1024), Nd4j.getNativeOps().getDeviceTotalMemory(0) / (1024 * 1024));

        for (int i = 0; i < 100; i++) {
            input = Nd4j.randn(DataType.FLOAT, 1, 1, hidden);
            sd.outputDirect(Map.of("x", input), "out");
            if (i % 25 == 0) {
                Nd4j.getExecutioner().commit();
                long free = Nd4j.getNativeOps().getDeviceFreeMemory(0);
                log.info("Step {}: free={} MB, growth={} MB",
                        i, free / (1024 * 1024), (freeBefore - free) / (1024 * 1024));
            }
        }

        Nd4j.getExecutioner().commit();
        long freeAfter = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long totalGrowth = freeBefore - freeAfter;
        double perStep = totalGrowth / 100.0 / (1024.0 * 1024.0);

        log.info("RESULT: total growth={} MB, per-step={} MB",
                totalGrowth / (1024 * 1024), String.format("%.2f", perStep));

        assertTrue(perStep < 5.0,
                "Per-step leak: " + String.format("%.2f MB (max 5.0)", perStep));

        sd.close();
    }
}
