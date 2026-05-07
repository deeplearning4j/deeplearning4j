package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation test: cuBLAS matmul determinism under CUDA graph capture vs. direct execution.
 *
 * The Qwen lm_head is [1,1024] x [1024,151936] -> [1,151936]. Tiny FP differences
 * in that 151936-wide logit vector flip argmax, causing token divergence.
 *
 * This test runs the EXACT same matmul both ways and compares bitwise.
 * If this fails, the cuBLAS algorithm selected during graph capture differs from
 * what runs in live execution — the deterministic cuBLAS setup is broken.
 */
@Slf4j
public class CudaGraphMatmulDeterminismTest {

    /**
     * Raw Nd4j.mmul test — no SameDiff, no DSP, just pure cuBLAS.
     * If THIS diverges between runs, cuBLAS itself is non-deterministic.
     */
    @Test
    public void testRawMmulDeterminism() {
        int hiddenDim = 1024;
        int vocabSize = 151936;
        int numRuns = 10;

        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        INDArray weight = Nd4j.randn(DataType.FLOAT, hiddenDim, vocabSize);

        INDArray reference = input.mmul(weight);
        int refArgmax = Nd4j.argMax(reference, 1).getInt(0);
        log.info("Reference argmax: {} (logit={})", refArgmax, reference.getFloat(0, refArgmax));

        int mismatches = 0;
        for (int i = 0; i < numRuns; i++) {
            INDArray result = input.mmul(weight);
            int argmax = Nd4j.argMax(result, 1).getInt(0);

            INDArray diff = Transforms.abs(reference.sub(result));
            float maxDiff = diff.maxNumber().floatValue();
            if (argmax != refArgmax) {
                log.warn("Run {}: argmax {} != reference {} (maxDiff={})",
                        i, argmax, refArgmax, maxDiff);
                mismatches++;
            } else if (maxDiff > 0) {
                log.info("Run {}: argmax matches but not bitwise identical (maxDiff={})",
                        i, maxDiff);
            } else {
                log.info("Run {}: BITWISE identical", i);
            }
            result.close();
            diff.close();
        }

        reference.close();
        input.close();
        weight.close();
        assertEquals(0, mismatches,
                String.format("Raw mmul non-deterministic: %d/%d runs had different argmax",
                        mismatches, numRuns));
    }

    /**
     * Test at FP16 — the actual compute path in the model uses FP16 weights.
     */
    @Test
    public void testLmHeadFp16Determinism() {
        int hiddenDim = 1024;
        int vocabSize = 151936;
        int numRuns = 10;

        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.randn(DataType.HALF, 1, hiddenDim);
        INDArray weight = Nd4j.randn(DataType.HALF, hiddenDim, vocabSize);

        INDArray reference = input.mmul(weight);
        int refArgmax = Nd4j.argMax(reference, 1).getInt(0);
        log.info("FP16 reference argmax: {} (logit={})", refArgmax, reference.getFloat(0, refArgmax));

        int mismatches = 0;
        for (int i = 0; i < numRuns; i++) {
            INDArray result = input.mmul(weight);
            int argmax = Nd4j.argMax(result, 1).getInt(0);

            INDArray diff = Transforms.abs(reference.sub(result));
            float maxDiff = diff.maxNumber().floatValue();
            if (argmax != refArgmax) {
                log.warn("FP16 Run {}: argmax {} != reference {} (maxDiff={})",
                        i, argmax, refArgmax, maxDiff);
                mismatches++;
            } else if (maxDiff > 0) {
                log.info("FP16 Run {}: argmax matches, not bitwise (maxDiff={})", i, maxDiff);
            } else {
                log.info("FP16 Run {}: BITWISE identical", i);
            }
            result.close();
            diff.close();
        }

        reference.close();
        input.close();
        weight.close();
        assertEquals(0, mismatches,
                String.format("FP16 mmul non-deterministic: %d/%d runs had different argmax",
                        mismatches, numRuns));
    }

    /**
     * SameDiff matmul: compare SLOT_BY_SLOT execution vs repeated runs.
     * This captures whether DSP adds non-determinism even without graph capture.
     */
    @Test
    public void testSameDiffMatmulSlotBySlot() {
        int hiddenDim = 1024;
        int vocabSize = 151936;
        int numExecs = 10;

        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        INDArray weight = Nd4j.randn(DataType.FLOAT, hiddenDim, vocabSize);

        // Ground truth: raw mmul
        INDArray rawResult = input.mmul(weight);
        int rawArgmax = Nd4j.argMax(rawResult, 1).getInt(0);
        log.info("Raw mmul argmax: {} (logit={})", rawArgmax, rawResult.getFloat(0, rawArgmax));

        // SameDiff execution
        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", DataType.FLOAT, -1, hiddenDim);
        SDVariable weightVar = sd.placeHolder("weight", DataType.FLOAT, hiddenDim, vocabSize);
        sd.mmul("output", inputVar, weightVar);

        int mismatches = 0;
        for (int i = 0; i < numExecs; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("input", input, "weight", weight),
                    Collections.singletonList("output"));
            INDArray out = result.get("output");
            int argmax = Nd4j.argMax(out, 1).getInt(0);

            INDArray diff = Transforms.abs(rawResult.sub(out));
            float maxDiff = diff.maxNumber().floatValue();
            if (argmax != rawArgmax) {
                log.warn("SameDiff exec {}: argmax {} != raw {} (maxDiff={})",
                        i, argmax, rawArgmax, maxDiff);
                mismatches++;
            } else if (maxDiff > 0) {
                log.info("SameDiff exec {}: argmax OK, not bitwise (maxDiff={})", i, maxDiff);
            } else {
                log.info("SameDiff exec {}: BITWISE identical to raw", i);
            }
            diff.close();
        }

        sd.close();
        rawResult.close();
        input.close();
        weight.close();
        assertEquals(0, mismatches,
                String.format("SameDiff SLOT_BY_SLOT matmul non-deterministic: %d/%d had different argmax",
                        mismatches, numExecs));
    }

    /**
     * THE KEY TEST: Same matmul run under DSP CUDA_GRAPHS capture/replay vs direct.
     * Runs the graph enough times to trigger capture, then compares replay output
     * to what direct cuBLAS would produce for the same input.
     */
    @Test
    public void testCudaGraphCaptureVsDirect() {
        int hiddenDim = 1024;
        int vocabSize = 151936;

        Nd4j.getRandom().setSeed(42);
        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hiddenDim);
        INDArray weight = Nd4j.randn(DataType.FLOAT, hiddenDim, vocabSize);

        // Ground truth: raw mmul (this is what cuBLAS produces without graph capture)
        INDArray rawResult = input.mmul(weight);
        int rawArgmax = Nd4j.argMax(rawResult, 1).getInt(0);
        log.info("Direct cuBLAS argmax: {} (logit={})", rawArgmax, rawResult.getFloat(0, rawArgmax));

        // SameDiff with CUDA_GRAPHS mode
        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", DataType.FLOAT, -1, hiddenDim);
        SDVariable weightVar = sd.placeHolder("weight", DataType.FLOAT, hiddenDim, vocabSize);
        sd.mmul("output", inputVar, weightVar);

        // Force CUDA_GRAPHS mode
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Execute enough times to go through: slot-by-slot -> freeze -> warmup -> capture -> replay
        int numExecs = 10;
        int firstMismatch = -1;
        for (int i = 0; i < numExecs; i++) {
            Map<String, INDArray> result = sd.output(
                    Map.of("input", input, "weight", weight),
                    Collections.singletonList("output"));
            INDArray out = result.get("output");
            int argmax = Nd4j.argMax(out, 1).getInt(0);

            INDArray diff = Transforms.abs(rawResult.sub(out));
            float maxDiff = diff.maxNumber().floatValue();

            String status = (argmax == rawArgmax) ?
                    (maxDiff == 0 ? "BITWISE" : String.format("OK (maxDiff=%.2e)", maxDiff)) :
                    String.format("MISMATCH argmax=%d vs raw=%d (maxDiff=%.2e)", argmax, rawArgmax, maxDiff);
            log.info("CUDA_GRAPHS exec {}: {}", i, status);

            if (argmax != rawArgmax && firstMismatch < 0) {
                firstMismatch = i;
                log.error("FIRST MISMATCH at exec {}: graph argmax={} vs direct argmax={}", i, argmax, rawArgmax);
                log.error("  Direct logit at argmax position: {}", rawResult.getFloat(0, rawArgmax));
                log.error("  Graph logit at argmax position:  {}", out.getFloat(0, rawArgmax));
                log.error("  Graph logit at graph-argmax:     {}", out.getFloat(0, argmax));
                log.error("  Diff at raw argmax:              {}", rawResult.getFloat(0, rawArgmax) - out.getFloat(0, rawArgmax));
                log.error("  Max diff anywhere:               {}", maxDiff);
            }
            diff.close();
        }

        sd.close();
        rawResult.close();
        input.close();
        weight.close();

        assertEquals(-1, firstMismatch,
                String.format("CUDA_GRAPHS argmax diverged from direct cuBLAS at exec %d", firstMismatch));
    }
}
