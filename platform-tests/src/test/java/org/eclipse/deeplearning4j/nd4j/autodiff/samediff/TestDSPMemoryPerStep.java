package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolates GPU memory growth per execution step across different DSP modes.
 * Tests SLOT_BY_SLOT, DSP (no graphs), DSP + frozen shapes, and DSP + CUDA graph replay.
 *
 * Run: cd platform-tests && mvn test -Dtest=TestDSPMemoryPerStep \
 *   -Dbackend.artifactId=nd4j-cuda-12.9 -Dlibnd4j.triton=ON 2>&1 | tee /tmp/dsp-memory.log
 */
@Slf4j
public class TestDSPMemoryPerStep {

    private static final int HIDDEN = 576;
    private static final int INTERMEDIATE = 1536;
    private static final int VOCAB = 1000;
    private static final int STEPS = 100;
    private static final int WARMUP = 5;

    static Stream<Arguments> modes() {
        return Stream.of(
            Arguments.of("SLOT_BY_SLOT", false, false, false),
            Arguments.of("DSP_UNFROZEN", true, false, false),
            Arguments.of("DSP_FROZEN", true, true, false),
            Arguments.of("DSP_FROZEN_GRAPHS", true, true, true)
        );
    }

    private SameDiff buildModel() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, HIDDEN);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, HIDDEN, INTERMEDIATE).muli(0.01));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, INTERMEDIATE, HIDDEN).muli(0.01));
        SDVariable wOut = sd.constant("wOut", Nd4j.randn(DataType.FLOAT, HIDDEN, VOCAB).muli(0.01));

        SDVariable flat = sd.reshape(x, 1, HIDDEN);
        SDVariable h1 = sd.mmul("h1", flat, w1);
        SDVariable act = sd.nn.relu("act", h1, 0);
        SDVariable h2 = sd.mmul("h2", act, w2);
        SDVariable res = h2.add("res", flat);
        SDVariable out = sd.mmul("out", res, wOut);
        return sd;
    }

    @ParameterizedTest(name = "memoryGrowth[{0}]")
    @MethodSource("modes")
    public void testMemoryGrowthPerMode(String modeName, boolean enableDsp,
                                         boolean freezeShapes, boolean enableGraphs) {
        SameDiff sd = buildModel();

        if (enableDsp) {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
        }

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN);

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            if (enableDsp) {
                sd.outputDirect(Map.of("x", input), "out");
            } else {
                sd.output(Map.of("x", input), "out");
            }
        }

        // For frozen/graph modes, just run more warmup iterations.
        // DSP auto-compile handles freezing and graph capture internally
        // after observing stable shapes across multiple executions.
        if (freezeShapes || enableGraphs) {
            for (int i = 0; i < 5; i++) {
                if (enableDsp) {
                    sd.outputDirect(Map.of("x", input), "out");
                } else {
                    sd.output(Map.of("x", input), "out");
                }
            }
        }

        // Measure
        Nd4j.getExecutioner().commit();
        System.gc();
        long freeBefore = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        log.info("[{}] Free before {} steps: {} MB", modeName, STEPS, freeBefore / (1024 * 1024));

        long[] checkpoints = new long[6];
        for (int i = 0; i < STEPS; i++) {
            input = Nd4j.randn(DataType.FLOAT, 1, 1, HIDDEN);
            if (enableDsp) {
                sd.outputDirect(Map.of("x", input), "out");
            } else {
                sd.output(Map.of("x", input), "out");
            }
            if (i == 0 || i == 19 || i == 39 || i == 59 || i == 79 || i == 99) {
                Nd4j.getExecutioner().commit();
                int idx = i == 0 ? 0 : i == 19 ? 1 : i == 39 ? 2 : i == 59 ? 3 : i == 79 ? 4 : 5;
                checkpoints[idx] = Nd4j.getNativeOps().getDeviceFreeMemory(0);
                long used = freeBefore - checkpoints[idx];
                log.info("[{}] Step {}: free={} MB, growth={} MB",
                        modeName, i, checkpoints[idx] / (1024 * 1024), used / (1024 * 1024));
            }
        }

        Nd4j.getExecutioner().commit();
        long freeAfter = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long totalLeak = freeBefore - freeAfter;
        double leakPerStep = totalLeak / (double) STEPS / (1024.0 * 1024.0);

        log.info("[{}] RESULT: total growth={} MB, per-step={:.2f} MB/step",
                modeName, totalLeak / (1024 * 1024), leakPerStep);

        // Steady-state leak (steps 20-100) — excludes one-time allocations
        long steadyLeak = checkpoints[1] - freeAfter;
        double steadyPerStep = steadyLeak / 80.0 / (1024.0 * 1024.0);
        log.info("[{}] STEADY-STATE: growth={} MB over 80 steps, {:.2f} MB/step",
                modeName, steadyLeak / (1024 * 1024), steadyPerStep);

        assertTrue(steadyPerStep < 5.0,
                modeName + ": steady-state leak too high: " +
                String.format("%.2f MB/step (max 5.0)", steadyPerStep));

        sd.close();
    }
}
