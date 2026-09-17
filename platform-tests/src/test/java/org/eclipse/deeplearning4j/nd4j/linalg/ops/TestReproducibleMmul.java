/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizationHelper;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.AlgebraicOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.AttentionFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.HorizontalFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.LinearFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.MatMulChainOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.NormalizationFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Independent ordered-FMA oracle, native ABI and isolated graph-rewrite contracts. */
@NativeTag
public class TestReproducibleMmul extends BaseNd4jTestWithBackends {
    @Override
    public char ordering() { return 'c'; }

    public static Stream<Arguments> storageConfigs() {
        return configs().flatMap(a -> Stream.of(DataType.FLOAT, DataType.HALF, DataType.BFLOAT16, DataType.DOUBLE)
                .map(t -> Arguments.of(a.get()[0], t)));
    }

    private static Mmul serial(INDArray a, INDArray b, INDArray out, double alpha, double beta,
                               boolean ta, boolean tb, boolean tz) {
        return new Mmul(a, b, out, alpha, beta, MMulTranspose.builder()
                .transposeA(ta).transposeB(tb).transposeResult(tz).build(), Mmul.Arithmetic.SERIAL_FMA);
    }

    private static SDVariable mm(SameDiff sd, SDVariable a, SDVariable b, boolean explicit) {
        return new Mmul(sd, a, b, MMulTranspose.allFalse(),
                explicit ? Mmul.Arithmetic.SERIAL_FMA : Mmul.Arithmetic.LEGACY).outputVariable();
    }

    private static SameDiffOp producer(SameDiff sd, SDVariable v) {
        return sd.getOps().get(sd.getVariables().get(v.name()).getOutputOfOp());
    }

    private static boolean apply(SameDiff sd, Optimizer pass, SDVariable anchor) {
        OptimizationHelper helper = new OptimizationHelper(sd, new Properties());
        helper.initializeCaches(sd);
        return pass.checkAndApply(sd, helper, producer(sd, anchor), sd.getConstantArrays(), sd.getVariablesArrays());
    }

    private static long count(SameDiff sd, String opName) {
        return sd.getOps().values().stream().filter(o -> opName.equals(o.getOp().opName())).count();
    }

    private static void assertBits(INDArray expected, INDArray actual, String message) {
        assertArrayEquals(expected.shape(), actual.shape(), message);
        assertEquals(expected.dataType(), actual.dataType(), message);
        for (long i = 0; i < expected.length(); i++) {
            long remaining = i;
            long[] coord = new long[expected.rank()];
            for (int d = coord.length - 1; d >= 0; d--) {
                coord[d] = remaining % expected.size(d);
                remaining /= expected.size(d);
            }
            assertDoubleBits(expected.getDouble(coord), actual.getDouble(coord),
                    message + " dtype=" + actual.dataType() + " coord=" + Arrays.toString(coord));
        }
    }

    private static void assertDoubleBits(double expected, double actual, String message) {
        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual),
                () -> message + " expected=" + expected + " (" + Double.toHexString(expected)
                        + ") actual=" + actual + " (" + Double.toHexString(actual) + ")");
    }

    private static void assertNegativeZero(INDArray actual, DataType dtype, long[] shape, String message) {
        assertEquals(dtype, actual.dataType(), message);
        assertArrayEquals(shape, actual.shape(), message);
        for (long i = 0; i < actual.length(); i++) {
            assertDoubleBits(-0.0, actual.getDouble(i), message + " dtype=" + dtype + " element=" + i);
        }
    }

    // Independent IEEE round-to-nearest, ties-to-even final store. Scaling a FLOAT
    // significand in DOUBLE is exact, and rint rounds the retained integer significand
    // to even. The minimum spacing includes storage subnormals, not a flush-to-zero.
    // Do not delegate rounding to Nd4j.scalar/putScalar/castTo: their host conversions
    // need not have the CUDA store's rounding policy. Return an exactly representable
    // storage value so putScalar performs no further rounding.
    private static double roundToStorage(double value, DataType dtype) {
        if (dtype == DataType.DOUBLE) return value;
        float f = (float) value;
        if (dtype == DataType.FLOAT) return f;
        if (dtype != DataType.HALF && dtype != DataType.BFLOAT16) {
            throw new IllegalArgumentException("Unsupported oracle storage dtype: " + dtype);
        }
        if (!Float.isFinite(f) || f == 0.0f) return f; // Preserve infinities, NaN and the sign of zero.
        int fractionBits = dtype == DataType.HALF ? 10 : 7;
        int minExponent = dtype == DataType.HALF ? -14 : -126;
        int maxExponent = dtype == DataType.HALF ? 15 : 127;
        int spacingExponent = Math.max(Math.getExponent(f), minExponent) - fractionBits;
        double rounded = Math.scalb(Math.rint(Math.scalb(Math.abs((double) f), -spacingExponent)),
                spacingExponent);
        if (rounded >= Math.scalb(1.0, maxExponent + 1)) rounded = Double.POSITIVE_INFINITY;
        return Math.copySign(rounded, f);
    }

    @Test
    public void storageOracleExactPatterns() {
        // Independent literal FLOAT bit patterns -> exact representable storage values.
        // Include both tie parities/signs, exponent carry, gradual underflow, overflow
        // and signed zero; none of these fixtures use NDArray conversion as a reference.
        int[] halfBits = {0x00000000, 0x80000000, 0x3f801000, 0x3f803000,
                0xbf801000, 0xbf803000, 0x3ffff000, 0x33000000, 0xb3000000,
                0x33000001, 0x33800000, 0x33c00000, 0x387fe000,
                0x477fe000, 0x477ff000, 0xc77ff000, 0x7f800000, 0xff800000};
        double[] halfValues = {0.0, -0.0, 1.0, 0x1.008p0, -1.0, -0x1.008p0,
                2.0, 0.0, -0.0, 0x1.0p-24, 0x1.0p-24, 0x1.0p-23, 0x1.0p-14,
                65504.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        int[] bf16Bits = {0x00000000, 0x80000000, 0x3f808000, 0x3f818000,
                0xbf808000, 0xbf818000, 0x3fff8000, 0x00008000, 0x80008000,
                0x00008001, 0x00010000, 0x00018000, 0x007f8000,
                0x7f7f0000, 0x7f7f8000, 0xff7f8000, 0x7f800000, 0xff800000};
        double[] bf16Values = {0.0, -0.0, 1.0, 0x1.04p0, -1.0, -0x1.04p0,
                2.0, 0.0, -0.0, 0x1.0p-133, 0x1.0p-133, 0x1.0p-132, 0x1.0p-126,
                0x1.fep127, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        for (int i = 0; i < halfBits.length; i++) {
            assertDoubleBits(halfValues[i], roundToStorage(Float.intBitsToFloat(halfBits[i]), DataType.HALF),
                    "HALF inputBits=0x" + Integer.toHexString(halfBits[i]));
        }
        for (int i = 0; i < bf16Bits.length; i++) {
            assertDoubleBits(bf16Values[i], roundToStorage(Float.intBitsToFloat(bf16Bits[i]), DataType.BFLOAT16),
                    "BFLOAT16 inputBits=0x" + Integer.toHexString(bf16Bits[i]));
        }
        for (DataType dtype : new DataType[]{DataType.HALF, DataType.BFLOAT16}) {
            assertTrue(Double.isNaN(roundToStorage(Float.intBitsToFloat(0x7f800001), dtype)), dtype.toString());
        }
        // The original W1 K=1 BF16 failure: truncation gave 0.99609375 instead of 1.0.
        assertDoubleBits(1.0, roundToStorage(1.0078125f * 0.9921875f, DataType.BFLOAT16),
                "BFLOAT16 near-one product");
    }

    // All values are read after conversion to the tested storage dtype. HALF/BF16 use a
    // FLOAT recurrence, not a DOUBLE dot followed by a cast. No native matmul is the oracle.
    private static double dot(INDArray a, INDArray b, long batch, long row, long col, long k,
                              boolean ta, boolean tb, double alpha, double beta, double old) {
        double sumD = 0.0;
        float sumF = 0.0f;
        for (long p = 0; p < k; p++) {
            double x = operand(a, batch, ta ? p : row, ta ? row : p);
            double y = operand(b, batch, tb ? col : p, tb ? p : col);
            if (a.dataType() == DataType.DOUBLE) sumD = Math.fma(x, y, sumD);
            else sumF = Math.fma((float) x, (float) y, sumF);
        }
        if (a.dataType() == DataType.DOUBLE) {
            double result = alpha * sumD;
            return beta == 0.0 ? result : Math.fma(beta, old, result);
        }
        float result = (float) alpha * sumF;
        return roundToStorage(beta == 0.0 ? result : Math.fma((float) beta, (float) old, result), a.dataType());
    }

    private static double operand(INDArray a, long batch, long row, long col) {
        if (a.rank() == 1) return a.getDouble(Math.max(row, col));
        if (a.rank() == 2) return a.getDouble(row, col);
        long[] coord = new long[a.rank()];
        coord[coord.length - 2] = row;
        coord[coord.length - 1] = col;
        for (int d = coord.length - 3; d >= 0; d--) {
            coord[d] = batch % a.size(d);
            batch /= a.size(d);
        }
        return a.getDouble(coord);
    }

    private static void fill(INDArray a, int salt) {
        double[] values = {1.0078125, 0.9921875, -1, 0.00006103515625, 16, -16, .333984375};
        for (long i = 0; i < a.length(); i++) a.putScalar(i, values[(int) ((i + salt) % values.length)]);
    }

    @ParameterizedTest(name = "ordered W1/W5 {0}/{1}")
    @MethodSource("storageConfigs")
    public void windowAndOracle(Nd4jBackend backend, DataType dtype) {
        for (int k : new int[]{1, 3, 31, 33, 257, 1025}) {
            try (INDArray one = Nd4j.create(dtype, 1, k);
                 INDArray five = Nd4j.create(dtype, 5, k);
                 INDArray weights = Nd4j.create(dtype, k, 7);
                 INDArray expected = Nd4j.create(dtype, 1, 7)) {
                fill(one, 0);
                fill(weights, 2);
                for (int r = 0; r < 5; r++) for (int p = 0; p < k; p++) five.putScalar(r, p, one.getDouble(0, p));
                for (int n = 0; n < 7; n++) expected.putScalar(0, n, dot(one, weights, 0, 0, n, k, false, false, 1, 0, Double.NaN));
                try (INDArray out1 = Nd4j.exec(serial(one, weights, null, 1, 0, false, false, false))[0];
                     INDArray out5 = Nd4j.exec(serial(five, weights, null, 1, 0, false, false, false))[0]) {
                    assertBits(expected, out1, "W1 K=" + k);
                    assertEquals(dtype, out5.dataType(), "W5 dtype K=" + k);
                    assertArrayEquals(new long[]{5, 7}, out5.shape(), "W5 shape K=" + k);
                    for (int r = 0; r < 5; r++) for (int n = 0; n < 7; n++) {
                        assertDoubleBits(expected.getDouble(0, n), out5.getDouble(r, n),
                                "W5 dtype=" + dtype + " row=" + r + " col=" + n + " K=" + k);
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "compiled SERIAL_FMA dynamic W1/W5 {0}/{1}")
    @MethodSource("storageConfigs")
    public void tritonDynamicWindowsAndIeeeOracle(Nd4jBackend backend, DataType dtype) {
        // CPU is covered by the native oracle tests above, not by a pretend compiled pass.
        // On CUDA, absent Triton is a failure, not an assumption/quiet native substitution.
        assumeTrue(Nd4j.backends().isCudaAvailable(), "Compiled Triton contract requires CUDA");
        assertTrue(Nd4j.getNativeOps().isTritonAvailable(), "CUDA coverage requires a live Triton backend");
        boolean previousDsp = InferenceSession.isDynamicShapePlanEnabled();
        boolean previousCompileAll = Nd4j.getEnvironment().tritonCompileAll();
        try {
            InferenceSession.setDynamicShapePlanEnabled(true);
            Nd4j.getEnvironment().setTritonCompileAll(true);
            for (boolean transposeB : new boolean[]{false, true}) {
                exerciseCompiledWindows(dtype, transposeB);
            }
        } finally {
            Nd4j.getEnvironment().setTritonCompileAll(previousCompileAll);
            InferenceSession.setDynamicShapePlanEnabled(previousDsp);
        }
    }

    private static void exerciseCompiledWindows(DataType dtype, boolean transposeB) {
        final int k = 33, n = 7;
        int br = transposeB ? n : k, bc = transposeB ? k : n;
        // Keep owners alive until the graph closes. Both operands have a nonzero base
        // and positive, non-unit strides; no dup/materialization hides input indexing.
        // C-order owners also keep native shape inference's result C-order, which is
        // required by the current compiled contract (F-order output is not admitted).
        try (INDArray oneOwner = Nd4j.create(dtype, new long[]{3, 2 * k + 2}, 'c');
             INDArray fiveOwner = Nd4j.create(dtype, new long[]{7, 2 * k + 2}, 'c');
             INDArray weightOwner = Nd4j.create(dtype, new long[]{2 * br + 2, 2 * bc + 2}, 'c');
             INDArray one = oneOwner.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 2, 2 * k + 1));
             INDArray five = fiveOwner.get(NDArrayIndex.interval(1, 6), NDArrayIndex.interval(1, 2, 2 * k + 1));
             INDArray weights = weightOwner.get(NDArrayIndex.interval(1, 2, 2 * br + 1),
                     NDArrayIndex.interval(1, 2, 2 * bc + 1));
             SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", dtype, -1, k);
            SDVariable w = sd.placeHolder("w", dtype, br, bc);
            // The alpha/beta SameDiff constructor exposes the three transpose arguments;
            // append the public fourth-argument ABI for SERIAL_FMA. Negative alpha makes
            // the signed-zero epilogue observable, not just equality of zero magnitudes.
            Mmul op = new Mmul(sd, x, w, -1.0, 0.0, false, transposeB);
            op.addIArgument(1);
            op.outputVariable().rename("out");
            sd.setOutputs("out");
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            // No forced GraphExecutionMode, manual freeze or cache clearing: each shape
            // warms up normally, then is revisited through the same shape-keyed cache.
            for (int width : new int[]{1, 5, 1, 5}) {
                INDArray input = width == 1 ? one : five;
                for (int step = 0; step < 12; step++) {
                    fillCompiledOperands(input, weights, transposeB, step % 4);
                    String context = dtype + " W=" + width + " transB=" + transposeB + " step=" + step;
                    try (INDArray expected = Nd4j.create(dtype, new long[]{width, n}, 'c');
                         INDArray nativeOutput = Nd4j.create(dtype, new long[]{width, n}, 'c')) {
                        // Signed zero is checked directly below, independently of host storage writes.
                        if (step % 4 != 3) {
                            for (int r = 0; r < width; r++) for (int c = 0; c < n; c++) {
                                expected.putScalar(r, c, dot(input, weights, 0, r, c, k,
                                        false, transposeB, -1, 0, Double.NaN));
                            }
                        }
                        Nd4j.exec(serial(input, weights, nativeOutput, -1, 0, false, transposeB, false));
                        if (step % 4 == 3) {
                            assertNegativeZero(nativeOutput, dtype, new long[]{width, n}, "native " + context);
                        } else {
                            assertBits(expected, nativeOutput, "native " + context);
                        }
                        INDArray actual = sd.output(Map.of("x", input, "w", weights), "out").get("out");
                        assertEquals('c', actual.ordering(), "compiled output admission " + context);
                        if (step % 4 == 3) {
                            assertNegativeZero(actual, dtype, new long[]{width, n}, "DSP " + context);
                        } else {
                            assertBits(expected, actual, "DSP " + context);
                        }
                        if (sd.dsp().isCompiled()) {
                            assertEquals(0, sd.dsp().lastExecSegmentsFailed(), context);
                        }
                        // Every numerical pattern must run after normal warmup, not
                        // merely the last (signed-zero) pattern before an audit check.
                        if (step >= 8) assertTritonExecution(sd.dsp(), context);
                    }
                }
                assertTritonExecution(sd.dsp(), dtype + " W=" + width + " transB=" + transposeB);
            }
        }
    }

    private static void fillCompiledOperands(INDArray a, INDArray b, boolean transposeB, int pattern) {
        DataType dtype = a.dataType();
        double epsilon = Math.scalb(1.0, dtype == DataType.DOUBLE ? -27 : dtype == DataType.FLOAT ? -13
                : dtype == DataType.HALF ? -10 : -7);
        // FLOAT/DOUBLE test gradual underflow and a halfway FMA. HALF tests an exactly
        // representable storage subnormal; BF16 uses its minimum normal to avoid making
        // this a host/device storage-conversion-underflow test instead of an FMA test.
        double tiny = dtype == DataType.DOUBLE ? Double.MIN_VALUE : dtype == DataType.FLOAT ? Float.MIN_VALUE
                : Math.scalb(1.0, dtype == DataType.HALF ? -24 : -126);
        for (int r = 0; r < a.size(0); r++) for (int p = 0; p < a.size(1); p++) {
            double value = pattern == 0 ? (p % 3 == 0 ? 1.0078125 : p % 3 == 1 ? -1.0 : .00006103515625)
                    : pattern == 1 ? (p == 0 ? -1 : p == 1 ? 1 + epsilon : 0)
                    : pattern == 2 ? (p < 2 ? tiny : 0) : -0.0;
            a.putScalar(r, p, value);
        }
        int k = (int) a.size(1), n = (int) b.size(transposeB ? 0 : 1);
        for (int p = 0; p < k; p++) for (int c = 0; c < n; c++) {
            double value = pattern == 0 ? (c + 1) * .125
                    : pattern == 1 ? (p == 1 ? 1 - epsilon : 1)
                    : pattern == 2 ? (p == 1 ? .5 : 1) : 1;
            b.putScalar(transposeB ? c : p, transposeB ? p : c, value);
        }
    }

    private static void assertTritonExecution(DspHandle handle, String context) {
        assertTrue(handle.isCompiled(), context);
        assertTrue(handle.lifecycleSnapshot().isShapesFrozenOrReplaying(), "never froze: " + context);
        assertTrue(handle.frozenExecCount() > 0, "no post-freeze execution: " + context);
        assertTrue(handle.numSegments() > 0, context);
        assertEquals(0, handle.lastExecSegmentsFailed(), context);
        boolean tritonExecuted = false;
        for (int i = 0; i < handle.numSegments(); i++) {
            String audit = handle.segmentCompilationAudit(i);
            JsonObject segment = new JsonParser().parse(audit).getAsJsonObject();
            assertFalse(segment.get("compilationFailed").getAsBoolean(), context + " " + audit);
            assertFalse(handle.isSegmentCaptureFailed(i), context + " " + audit);
            if (handle.segmentCompiledBackend(i).contains("Triton")) {
                assertTrue(segment.get("compiledByBackend").getAsString().contains("Triton"), audit);
                assertTrue(segment.get("capturable").getAsBoolean(), audit);
                assertTrue(segment.get("executionCount").getAsLong() > 0, audit);
                tritonExecuted = true;
            }
        }
        // This graph has only the SERIAL_FMA matmul: another compiled op cannot satisfy it.
        assertTrue(tritonExecuted, "SERIAL_FMA must actually compile and execute in Triton: " + context);
    }

    @ParameterizedTest(name = "FMA rounding and IEEE boundaries {0}/{1}")
    @MethodSource("storageConfigs")
    public void roundingBoundaries(Nd4jBackend backend, DataType dtype) {
        double epsilon = Math.scalb(1.0, dtype == DataType.DOUBLE ? -27 : dtype == DataType.FLOAT ? -13
                : dtype == DataType.HALF ? -10 : -7);
        try (INDArray a = Nd4j.create(dtype, 1, 2); INDArray b = Nd4j.create(dtype, 2, 1);
             INDArray c = Nd4j.create(dtype, 1, 1); INDArray expected = Nd4j.create(dtype, 1, 1)) {
            a.putScalar(0, 0, -1); a.putScalar(0, 1, 1 + epsilon);
            b.putScalar(0, 0, 1); b.putScalar(1, 0, 1 - epsilon);
            expected.putScalar(0, roundToStorage(-epsilon * epsilon, dtype));
            Nd4j.exec(serial(a, b, c, 1, 0, false, false, false));
            assertBits(expected, c, "FMA residual rather than multiply-then-add");
            a.assign(-0.0); b.assign(1);
            Nd4j.exec(serial(a, b, c, -1, 0, false, false, false));
            // assign(-0.0) may take an unsigned zero-fill path for HALF. The expected
            // sign is the IEEE epilogue contract, not another NDArray's construction.
            assertNegativeZero(c, dtype, new long[]{1, 1},
                    "positive-zero initial accumulator followed by negative alpha");
            if (dtype == DataType.FLOAT || dtype == DataType.DOUBLE) {
                double minimum = dtype == DataType.FLOAT ? Float.MIN_VALUE : Double.MIN_VALUE;
                a.assign(minimum); b.putScalar(0, 0, 1); b.putScalar(1, 0, .5);
                expected.putScalar(0, 2 * minimum);
                Nd4j.exec(serial(a, b, c, 1, 0, false, false, false));
                assertBits(expected, c, "gradual underflow and ties-to-even");
            }
            a.assign(Double.POSITIVE_INFINITY); b.assign(1);
            c.assign(Double.NaN);
            Nd4j.exec(serial(a, b, c, 1, 0, false, false, false));
            assertEquals(Double.POSITIVE_INFINITY, c.getDouble(0));
            a.assign(Double.NaN);
            Nd4j.exec(serial(a, b, c, 1, 0, false, false, false));
            assertTrue(Double.isNaN(c.getDouble(0))); // NaN payload identity is not promised.
        }
    }

    @ParameterizedTest(name = "layouts/batches/transposes/epilogue {0}/{1}")
    @MethodSource("storageConfigs")
    public void matrixContract(Nd4jBackend backend, DataType dtype) {
        for (char order : new char[]{'c', 'f'}) for (int flags = 0; flags < 8; flags++) {
            boolean ta = (flags & 1) != 0, tb = (flags & 2) != 0, tz = (flags & 4) != 0;
            // Equal-rank batches and both advertised ND/2D broadcast directions.
            for (int batching = 0; batching < 7; batching++) {
                long m = 3, k = 33, n = 5, batches = batching == 0 ? 1 : 2;
                long[] as = batching == 1 || batching == 2
                        ? new long[]{2, ta ? k : m, ta ? m : k} : new long[]{ta ? k : m, ta ? m : k};
                long[] bs = batching == 1 || batching == 3
                        ? new long[]{2, tb ? n : k, tb ? k : n} : new long[]{tb ? n : k, tb ? k : n};
                long[] os = batching == 0 ? new long[]{tz ? n : m, tz ? m : n}
                        : new long[]{2, tz ? n : m, tz ? m : n};
                if (batching >= 4) {
                    batches = 6;
                    if (batching != 6) as = new long[]{2, 3, ta ? k : m, ta ? m : k};
                    if (batching != 5) bs = new long[]{2, 3, tb ? n : k, tb ? k : n};
                    os = new long[]{2, 3, tz ? n : m, tz ? m : n};
                }
                for (double beta : new double[]{0, -0.0, .375}) {
                    try (INDArray a = Nd4j.create(dtype, as, order); INDArray b = Nd4j.create(dtype, bs, order);
                         INDArray out = Nd4j.create(dtype, os, order).assign(beta == 0 ? Double.NaN : 1.25);
                         INDArray expected = Nd4j.create(dtype, os, order)) {
                        fill(a, 0); fill(b, 3);
                        for (int batch = 0; batch < batches; batch++) for (int r = 0; r < m; r++) for (int c = 0; c < n; c++) {
                            long[] coord = batching == 0 ? new long[]{tz ? c : r, tz ? r : c}
                                    : batching < 4 ? new long[]{batch, tz ? c : r, tz ? r : c}
                                    : new long[]{batch / 3, batch % 3, tz ? c : r, tz ? r : c};
                            expected.putScalar(coord, dot(a, b, batch, r, c, k, ta, tb, .7, beta, 1.25));
                        }
                        Nd4j.exec(serial(a, b, out, .7, beta, ta, tb, tz));
                        assertBits(expected, out, "flags=" + flags + " batch=" + batching + " beta=" + beta + " order=" + order);
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "offset stepped views {0}/{1}")
    @MethodSource("storageConfigs")
    public void viewsAndSentinels(Nd4jBackend backend, DataType dtype) {
        try (INDArray ap = Nd4j.create(dtype, 5, 70).assign(77);
             INDArray bp = Nd4j.create(dtype, 70, 9).assign(88);
             INDArray cp = Nd4j.create(dtype, new long[]{5, 12}, 'f').assign(99);
             INDArray a = ap.get(NDArrayIndex.interval(1, 4), NDArrayIndex.interval(2, 2, 68));
             INDArray b = bp.get(NDArrayIndex.interval(1, 2, 67), NDArrayIndex.interval(2, 7));
             INDArray c = cp.get(NDArrayIndex.interval(1, 4), NDArrayIndex.interval(1, 2, 11));
             INDArray expected = Nd4j.create(dtype, 3, 5)) {
            fill(a, 0); fill(b, 1);
            for (double beta : new double[]{0, .25}) {
                c.assign(beta == 0 ? Double.NaN : 2);
                for (int r = 0; r < 3; r++) for (int col = 0; col < 5; col++) {
                    expected.putScalar(r, col, dot(a, b, 0, r, col, 33, false, false, 1, beta, 2));
                }
                Nd4j.exec(serial(a, b, c, 1, beta, false, false, false));
                assertBits(expected, c, "strided output beta=" + beta);
                for (int r = 0; r < 5; r++) for (int col = 0; col < 12; col++) {
                    if (r >= 1 && r < 4 && col >= 1 && col < 11 && col % 2 == 1) continue;
                    assertEquals(99, cp.getDouble(r, col), 0, "untouched sentinel");
                }
            }
        }
    }

    @ParameterizedTest(name = "vectors and empty shape {0}/{1}")
    @MethodSource("storageConfigs")
    public void vectorsAndEmptyShape(Nd4jBackend backend, DataType dtype) {
        try (INDArray a = Nd4j.create(dtype, 33); INDArray b = Nd4j.create(dtype, 33);
             INDArray matrix = Nd4j.create(dtype, 33, 5)) {
            fill(a, 0); fill(b, 2); fill(matrix, 3);
            try (INDArray scalar = Nd4j.exec(serial(a, b, null, 1, 0, false, false, false))[0];
                 INDArray expected = Nd4j.create(dtype, new long[0])) {
                expected.putScalar(0, dot(a, b, 0, 0, 0, 33, false, false, 1, 0, 0));
                assertBits(expected, scalar, "vector dot");
            }
            try (INDArray out = Nd4j.exec(serial(a, matrix, null, 1, 0, false, false, false))[0];
                 INDArray expected = Nd4j.create(dtype, 5)) {
                for (int n = 0; n < 5; n++) expected.putScalar(n, dot(a, matrix, 0, 0, n, 33, false, false, 1, 0, 0));
                assertBits(expected, out, "vector matrix");
                try (INDArray transposed = matrix.transpose();
                     INDArray reverse = Nd4j.exec(serial(transposed, a, null, 1, 0, false, false, false))[0]) {
                    assertBits(expected, reverse, "matrix vector");
                }
            }
        }
        // Existing matmul semantics mark the inferred result empty for K=0; do not
        // invent a nonempty zero-filled output contract for this source batch.
        try (INDArray a = Nd4j.create(dtype, 2, 0); INDArray b = Nd4j.create(dtype, 0, 3)) {
            DynamicCustomOp op = DynamicCustomOp.builder("matmul").addInputs(a, b).addIntegerArguments(0, 0, 0, 1).build();
            long[] shapeInfo = op.calculateOutputShape().get(0).asLong();
            assertArrayEquals(new long[]{2, 3}, Shape.shape(shapeInfo));
            assertTrue(Shape.isEmpty(shapeInfo));
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void serializationAndIdentity(Nd4jBackend backend) throws IOException {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 2, 3);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 3, 4);
        SDVariable out = mm(sd, a, b, true).rename("out");
        sd.setOutputs(out.name());
        SameDiff restored = SameDiff.fromFlatBuffers(sd.asFlatBuffers(true));
        Mmul copy = (Mmul) restored.getVariableOutputOp("out");
        assertArrayEquals(new long[]{0, 0, 0, 1}, copy.iArgs());
        assertEquals(Mmul.Arithmetic.SERIAL_FMA, copy.arithmetic());
        Mmul legacy = new Mmul();
        legacy.addIArgument(0, 0, 0);
        Mmul explicit = new Mmul();
        explicit.addIArgument(0, 0, 0, 1);
        assertEquals(Mmul.Arithmetic.LEGACY, legacy.arithmetic());
        assertNotEquals(legacy, explicit);
        assertNotEquals(legacy.hashCode(), explicit.hashCode());
        Set<Mmul> identities = new HashSet<>(Arrays.asList(legacy, explicit));
        assertEquals(2, identities.size());
        assertThrows(IllegalStateException.class, () -> copy.doDiff(Collections.singletonList(out)));
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void invalidContracts(Nd4jBackend backend) {
        try (INDArray a = Nd4j.ones(DataType.FLOAT, 2, 2); INDArray b = a.dup();
             INDArray c = a.dup(); INDArray half = b.castTo(DataType.HALF); INDArray integer = b.castTo(DataType.INT);
             INDArray wrongOutput = Nd4j.create(DataType.FLOAT, 2, 3);
             INDArray wrongK = Nd4j.create(DataType.FLOAT, 3, 2);
             INDArray scalar = Nd4j.scalar(1.0f)) {
            for (long flag : new long[]{-1, 2, Long.MAX_VALUE}) {
                assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("matmul")
                        .addInputs(a, b).addOutputs(c).addIntegerArguments(0, 0, 0, flag).build()));
            }
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, half, c, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, b, half, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(integer, integer, integer, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, b, a, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, b, b, 1, 0, false, false, false)));
            try (INDArray view = a.transpose()) {
                assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, b, view, 1, 0, false, false, false)));
            }
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, b, wrongOutput, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(a, wrongK, c, 1, 0, false, false, false)));
            assertThrows(RuntimeException.class, () -> Nd4j.exec(serial(scalar, scalar, null, 1, 0, false, false, false)));
            Mmul op = serial(a, b, c, 1, 0, false, false, false);
            assertThrows(IllegalStateException.class, () -> op.calculateOutputDataTypes(Arrays.asList(DataType.FLOAT, DataType.HALF)));
            assertThrows(IllegalStateException.class, () -> op.calculateOutputDataTypes(Arrays.asList(DataType.INT, DataType.INT)));
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void linearNormalizationAndScaleRewrites(Nd4jBackend backend) {
        for (boolean explicit : new boolean[]{false, true}) {
            for (int pattern = 0; pattern < 3; pattern++) {
                SameDiff sd = SameDiff.create();
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
                SDVariable w = sd.constant("w", Nd4j.ones(DataType.FLOAT, 4, 3));
                SDVariable input = pattern == 1
                        ? new RmsNorm(sd, x, sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 4)), 1e-5).outputVariable() : x;
                SDVariable matmul = mm(sd, input, w, explicit);
                SDVariable anchor;
                Optimizer pass;
                if (pattern == 0) {
                    anchor = matmul.add(sd.constant("bias", Nd4j.ones(DataType.FLOAT, 3)));
                    pass = new LinearFusionOptimizations.FuseMatMulWithAdd();
                } else if (pattern == 1) {
                    anchor = matmul;
                    pass = new NormalizationFusionOptimizations.FuseRMSNormLinearPattern();
                } else {
                    anchor = matmul.mul(sd.constant("scale", Nd4j.scalar(0.7f)));
                    pass = new AlgebraicOptimizations.ScalarIntoWeightFolding();
                }
                long[] originalArgs = ((Mmul) producer(sd, matmul).getOp()).iArgs().clone();
                assertEquals(!explicit, apply(sd, pass, anchor), "pattern=" + pattern + " explicit=" + explicit);
                if (explicit) {
                    assertArrayEquals(originalArgs, ((Mmul) producer(sd, matmul).getOp()).iArgs());
                    assertEquals("w", producer(sd, matmul).getInputsToOp().get(1));
                    assertEquals(1, count(sd, "matmul"));
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void horizontalMixedPolicies(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable protectedOut = mm(sd, x, sd.constant("wp", Nd4j.ones(DataType.FLOAT, 4, 3)), true);
        SDVariable legacy1 = mm(sd, x, sd.constant("w1", Nd4j.ones(DataType.FLOAT, 4, 3)), false);
        mm(sd, x, sd.constant("w2", Nd4j.ones(DataType.FLOAT, 4, 5)), false);
        assertFalse(apply(sd, new HorizontalFusionOptimizations.FuseParallelMatmuls(), protectedOut));
        assertTrue(apply(sd, new HorizontalFusionOptimizations.FuseParallelMatmuls(), legacy1));
        assertEquals(2, count(sd, "matmul"));
        assertEquals(Mmul.Arithmetic.SERIAL_FMA, ((Mmul) producer(sd, protectedOut).getOp()).arithmetic());
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void attentionBothMatmuls(Nd4jBackend backend) {
        for (int pattern = 0; pattern < 4; pattern++) for (int explicitMask = 0; explicitMask < 4; explicitMask++) {
            SameDiff sd = SameDiff.create();
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 2, 4);
            SDVariable key = sd.placeHolder("key", DataType.FLOAT, 1, 2, 4);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, 2, 4);
            SDVariable scores = new Mmul(sd, q, key, MMulTranspose.builder().transposeB(true).build(),
                    (explicitMask & 1) != 0 ? Mmul.Arithmetic.SERIAL_FMA : Mmul.Arithmetic.LEGACY).outputVariable();
            SDVariable masked = scores;
            if (pattern == 1) masked = scores.add(sd.constant("mask", Nd4j.createFromArray(new float[][]{
                    {0, Float.NEGATIVE_INFINITY}, {0, 0}})));
            if (pattern == 2) masked = scores.add(sd.placeHolder("mask", DataType.FLOAT, 1, 2, 2));
            SDVariable probs = sd.nn().softmax(masked, -1);
            SDVariable output = mm(sd, probs, v, (explicitMask & 2) != 0);
            Optimizer pass = pattern == 0 ? new AttentionFusionOptimizations.FuseManualAttentionPattern()
                    : pattern == 1 ? new AttentionFusionOptimizations.FuseAttentionWithCausalMask()
                    : pattern == 2 ? new AttentionFusionOptimizations.FuseAttentionWithMask()
                    : new AttentionFusionOptimizations.FuseLLaMAAttentionBlock();
            SDVariable anchor = pattern == 0 ? output : masked;
            if (pattern == 3) anchor = mm(sd, output, sd.constant("o_proj.weight", Nd4j.ones(DataType.FLOAT, 4, 4)), false);
            assertEquals(explicitMask == 0, apply(sd, pass, anchor),
                    "attention pattern=" + pattern + " explicitMask=" + explicitMask);
            if (explicitMask != 0) assertEquals(pattern == 3 ? 3 : 2, count(sd, "matmul"));
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void weightQuantizationPreservesOperandAncestors(Nd4jBackend backend) {
        for (int mode = 0; mode < 6; mode++) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable protectedWeight = sd.constant("protectedWeight", Nd4j.valueArrayOf(new long[]{32, 32}, 1.0001, DataType.FLOAT));
            SDVariable trainable = sd.var("trainable", Nd4j.ones(DataType.FLOAT, 32, 32));
            SDVariable ordinary = sd.constant("ordinary", Nd4j.ones(DataType.FLOAT, 32, 32));
            mm(sd, x, protectedWeight.castTo(DataType.DOUBLE).castTo(DataType.FLOAT), true);
            mm(sd, x, trainable, true);
            mm(sd, x, ordinary, false);
            mm(sd, x, protectedWeight, false); // shared storage cannot be mutated for the legacy user
            double original = sd.getConstantArrays().getArray("protectedWeight").getDouble(0);
            if (mode == 0) QuantizationOptimizations.QuantizeConstantsToFP16.quantizeAllToHalf(sd);
            if (mode == 1) QuantizationOptimizations.QuantizeConstantsToFP16.quantizeAllToBFloat16(sd);
            if (mode == 2) QuantizationOptimizations.QuantizeConstantsToINT8.quantizeAllConstants(sd);
            if (mode == 3) QuantizationOptimizations.QuantizeConstantsToINT8.quantizeAllConstantsWithScales(sd);
            if (mode == 4) QuantizationOptimizations.QuantizeConstantsToFP16.quantizeAllToType(sd, DataType.FLOAT8);
            if (mode == 5) QuantizationOptimizations.QuantizeConstantsToFP16.quantizeAllToType(sd, DataType.FLOAT8_E5M2);
            assertEquals(DataType.FLOAT, sd.getConstantArrays().getArray("protectedWeight").dataType());
            assertEquals(original, sd.getConstantArrays().getArray("protectedWeight").getDouble(0), 0);
            assertEquals(DataType.FLOAT, sd.getVariablesArrays().getArray("trainable").dataType());
            assertNotEquals(DataType.FLOAT, sd.getConstantArrays().getArray("ordinary").dataType());
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void activationQuantizationAndCastBoundaries(Nd4jBackend backend) {
        String property = QuantizationOptimizations.QuantizeActivationsInt8.PROP_ENABLE;
        String previous = System.getProperty(property);
        try {
            System.setProperty(property, "true");
            for (boolean explicit : new boolean[]{false, true}) {
                SameDiff sd = SameDiff.create();
                SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
                SDVariable w = sd.constant("w", Nd4j.ones(DataType.FLOAT, 4, 3));
                SDVariable out = mm(sd, x, w, explicit);
                Optimizer quantize = new QuantizationOptimizations.QuantizeActivationsInt8(
                        Collections.singletonMap("x", new double[]{-2, 2}));
                assertEquals(!explicit, apply(sd, quantize, out));
                assertEquals(explicit, "x".equals(producer(sd, out).getInputsToOp().get(0)));
            }
            // An ordinary matmul upstream is not an excuse to quantize explicit operands.
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable w = sd.constant("w", Nd4j.ones(DataType.FLOAT, 4, 4));
            SDVariable upstream = mm(sd, x, w, false);
            mm(sd, upstream, w, true);
            assertFalse(apply(sd, new QuantizationOptimizations.QuantizeActivationsInt8(
                    Collections.singletonMap("x", new double[]{-2, 2})), upstream));
        } finally {
            if (previous == null) System.clearProperty(property); else System.setProperty(property, previous);
        }
        for (DataType intermediate : new DataType[]{DataType.HALF, DataType.DOUBLE}) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable roundTrip = x.castTo(intermediate).castTo(DataType.FLOAT);
            SDVariable out = mm(sd, roundTrip, sd.constant("w", Nd4j.ones(DataType.FLOAT, 4, 3)), true);
            assertEquals(intermediate == DataType.DOUBLE,
                    apply(sd, new QuantizationOptimizations.RemoveRedundantCasts(), roundTrip));
            assertEquals(Mmul.Arithmetic.SERIAL_FMA, ((Mmul) producer(sd, out).getOp()).arithmetic());
            assertEquals(intermediate == DataType.DOUBLE ? 1 : 2, count(sd, "cast"));
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void constantChainPolicyAndNonMatrixTranspose(Nd4jBackend backend) {
        for (int explicitMask = 0; explicitMask < 4; explicitMask++) {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
            SDVariable first = mm(sd, x, sd.constant("w1", Nd4j.ones(DataType.FLOAT, 4, 3)), (explicitMask & 1) != 0);
            SDVariable second = mm(sd, first, sd.constant("w2", Nd4j.ones(DataType.FLOAT, 3, 5)), (explicitMask & 2) != 0);
            assertEquals(explicitMask == 0, apply(sd, new MatMulChainOptimizations.FoldConstantMatMulChain(), second));
            assertEquals(explicitMask == 0 ? 1 : 2, count(sd, "matmul"));
        }
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable transposed = sd.transpose(x); // [4,3,2], NOT a last-two-axis transpose
        SDVariable out = mm(sd, transposed, sd.constant("w", Nd4j.ones(DataType.FLOAT, 2, 5)), true);
        assertFalse(apply(sd, new MatMulChainOptimizations.TransposeMatMulFusion(), out));
        assertEquals(1, count(sd, "transpose"));
        assertArrayEquals(new long[]{0, 0, 0, 1}, ((Mmul) producer(sd, out).getOp()).iArgs());
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void optimizerAndTransposePreserveArguments(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 2);
        SDVariable w = sd.constant("w", Nd4j.ones(DataType.FLOAT, 4, 3));
        SDVariable out = mm(sd, sd.transpose(x), w, true).rename("out");
        List<OptimizerSet> passes = Arrays.asList(new MatMulChainOptimizations(), new HorizontalFusionOptimizations(),
                new LinearFusionOptimizations(), new NormalizationFusionOptimizations(),
                new AttentionFusionOptimizations(), new AlgebraicOptimizations());
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList(out.name()), passes);
        assertEquals(1, count(optimized, "matmul"));
        Mmul kept = (Mmul) optimized.getVariableOutputOp("out");
        assertArrayEquals(new long[]{1, 0, 0, 1}, kept.iArgs());
        assertEquals(0, count(optimized, "transpose"));
    }
}
