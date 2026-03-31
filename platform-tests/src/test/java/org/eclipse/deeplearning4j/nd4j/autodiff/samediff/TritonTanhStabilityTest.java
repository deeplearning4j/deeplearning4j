package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone tests for Triton-compiled tanh and GELU kernels with extreme input values.
 * These test numerical stability — the original Triton tanh emitter used
 * (exp(2x)-1)/(exp(2x)+1) which overflows for |x| > ~44, producing NaN.
 * The fix clamps input to [-10,10] before exp(2x).
 */
@Slf4j
public class TritonTanhStabilityTest {

    private static final double TOLERANCE = 1e-4;

    /**
     * Test tanh with values that would overflow the old formula.
     * tanh(50) = 1.0 exactly in float32, but exp(100) = Inf.
     */
    @Test
    public void testTanhLargePositiveValues() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1);
        SDVariable output = sd.math().tanh("output", input);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Values that cause exp(2x) overflow: 50, 100, 1000, Float.MAX_VALUE/2
        INDArray inputArr = Nd4j.create(new float[]{
                0.0f, 0.5f, 1.0f, 5.0f, 10.0f, 20.0f, 50.0f, 100.0f,
                -0.5f, -1.0f, -5.0f, -10.0f, -20.0f, -50.0f, -100.0f, 0.001f
        }, new int[]{1, 16});

        // Reference: compute via Nd4j (uses native stable tanh)
        INDArray reference = Nd4j.math().tanh(inputArr);

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "output");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute 3 times with frozen shapes
        for (int step = 0; step < 3; step++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(), "Step " + step + ": NaN in tanh output");
            assertFalse(actual.isInfinite().any(), "Step " + step + ": Inf in tanh output");

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("Step {}: maxDiff={}, output={}", step, maxDiff,
                    actual.data().asFloat());
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d: maxDiff=%.6f exceeds tolerance=%.6f\nref=%s\nact=%s",
                            step, maxDiff, TOLERANCE,
                            java.util.Arrays.toString(reference.data().asFloat()),
                            java.util.Arrays.toString(actual.data().asFloat())));
        }

        sd.close();
    }

    /**
     * Test GELU (tanh approximation) with large values.
     * GELU uses tanh internally: gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
     * For large x, the tanh argument can be huge → overflow in old formula.
     *
     * Note: SameDiff's built-in GELU uses the erf form, so this tests the
     * fused pattern (multiply chains that form GELU) via the tanh emitter.
     */
    @Test
    public void testTanhInGeluRange() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1);

        // Build the tanh-approximation GELU manually to exercise the tanh emitter:
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        SDVariable x3 = sd.math().pow(input, 3.0);
        SDVariable inner = input.add(x3.mul(0.044715));
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        SDVariable tanhArg = inner.mul(sqrt2OverPi);
        SDVariable tanhResult = sd.math().tanh(tanhArg);
        SDVariable output = input.mul(tanhResult.add(1.0)).mul(0.5);
        output.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Include values where x^3 * 0.044715 * sqrt(2/pi) > 44 (overflow threshold)
        // x = 10: tanh_arg ≈ sqrt(2/pi) * (10 + 0.044715*1000) ≈ 0.7979 * 54.715 ≈ 43.6
        // x = 11: tanh_arg ≈ 0.7979 * (11 + 0.044715*1331) ≈ 0.7979 * 70.5 ≈ 56.3 → overflow!
        INDArray inputArr = Nd4j.create(new float[]{
                0.0f, 1.0f, 2.0f, 5.0f, 8.0f, 10.0f, 11.0f, 15.0f,
                -1.0f, -2.0f, -5.0f, -8.0f, -10.0f, -11.0f, -15.0f, 3.0f
        }, new int[]{1, 16});

        // Reference GELU via manual computation
        float[] ref = new float[16];
        float[] in = inputArr.data().asFloat();
        for (int i = 0; i < 16; i++) {
            double x = in[i];
            double tArg = sqrt2OverPi * (x + 0.044715 * x * x * x);
            double t = Math.tanh(tArg);
            ref[i] = (float) (0.5 * x * (1.0 + t));
        }
        INDArray reference = Nd4j.create(ref, new int[]{1, 16});

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "output");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute with frozen shapes
        for (int step = 0; step < 3; step++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");
            assertFalse(actual.isNaN().any(),
                    "Step " + step + ": NaN in GELU output. Values: " +
                            java.util.Arrays.toString(actual.data().asFloat()));
            assertFalse(actual.isInfinite().any(),
                    "Step " + step + ": Inf in GELU output. Values: " +
                            java.util.Arrays.toString(actual.data().asFloat()));

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("GELU Step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("GELU Step %d: maxDiff=%.6f exceeds tolerance=%.6f",
                            step, maxDiff, TOLERANCE));
        }

        sd.close();
    }

    /**
     * Test standalone tanh with a larger tensor (realistic size).
     * Uses values sampled from N(0,5) which can produce values > 15.
     */
    @Test
    public void testTanhLargeTensorRandomValues() {
        SameDiff sd = SameDiff.create();
        int rows = 64, cols = 128;
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, rows, cols);
        SDVariable output = sd.math().tanh("output", input);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Random values with wide distribution (std=5) — many values > 10
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, rows, cols).mul(5.0);

        // Inject some extreme values
        inputArr.putScalar(0, 0, 50.0f);
        inputArr.putScalar(0, 1, -50.0f);
        inputArr.putScalar(0, 2, 100.0f);
        inputArr.putScalar(0, 3, -100.0f);
        inputArr.putScalar(0, 4, Float.MAX_VALUE / 2);
        inputArr.putScalar(0, 5, -Float.MAX_VALUE / 2);

        INDArray reference = Nd4j.math().tanh(inputArr);

        Map<String, INDArray> feed = Map.of("input", inputArr);

        // Warmup
        sd.outputDirect(feed, "output");

        // Compile with Triton
        try {
            sd.compileNativeDynamicShapePlan(List.of("output"),
                    GraphExecutionMode.TRITON, true);
        } catch (Exception e) {
            log.warn("Triton compilation not available: {}", e.getMessage());
            sd.close();
            return;
        }

        // Execute with frozen shapes
        for (int step = 0; step < 3; step++) {
            Map<String, INDArray> result = sd.outputDirect(feed, "output");
            INDArray actual = result.get("output");
            assertNotNull(actual, "Step " + step + " output is null");

            long nanCount = actual.isNaN().castTo(DataType.INT64).sumNumber().longValue();
            long infCount = actual.isInfinite().castTo(DataType.INT64).sumNumber().longValue();
            assertEquals(0, nanCount, "Step " + step + ": found " + nanCount + " NaN values in tanh output");
            assertEquals(0, infCount, "Step " + step + ": found " + infCount + " Inf values in tanh output");

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            log.info("Large tensor step {}: maxDiff={}, nanCount={}, infCount={}",
                    step, maxDiff, nanCount, infCount);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d: maxDiff=%.6f exceeds tolerance=%.6f",
                            step, maxDiff, TOLERANCE));
        }

        sd.close();
    }
}
