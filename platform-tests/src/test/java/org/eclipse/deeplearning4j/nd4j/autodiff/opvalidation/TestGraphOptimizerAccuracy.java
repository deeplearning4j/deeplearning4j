package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone graph tests that isolate specific patterns the GraphOptimizer
 * encounters in real vision encoder / transformer models.
 *
 * Each test constructs a targeted graph mimicking a specific facet of the
 * real model, runs it unoptimized for a reference, then applies a specific
 * optimizer pass and asserts the output values match numerically.
 *
 * These patterns are derived from the SmolDocling vision encoder architecture:
 *   patch_embedding(conv2d) → reshape → transpose → layernorm → attention → MLP
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestGraphOptimizerAccuracy \
 *     2>&1 | tee /tmp/optimizer-accuracy.log
 */
@Slf4j
@NativeTag
@Tag("samediff")
public class TestGraphOptimizerAccuracy {

    @BeforeEach
    public void clearProps() {
        System.clearProperty("nd4j.optimizer.fp16");
        System.clearProperty("nd4j.optimizer.bf16");
    }

    @AfterEach
    public void restoreProps() {
        System.clearProperty("nd4j.optimizer.fp16");
        System.clearProperty("nd4j.optimizer.bf16");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // FP16 quantization: conv2d weight quantization
    // Vision encoder starts with a patch embedding conv2d [3→768, k=16, s=16].
    // FP16 pass converts the 2D+ weight to HALF. The conv must handle mixed
    // HALF weight + FLOAT input correctly or output goes to zero/garbage.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFP16_Conv2dWeightQuantization() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        // Simulate patch embedding: [1,3,64,64] → conv2d(k=16,s=16) → [1,48,4,4]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, 64, 64);
        // SameDiff conv2d expects weights in [kH, kW, inC, outC] (YXIO) format
        // Large 4D weight that FP16 quantization will target (>= 1024 elements)
        INDArray weightArr = Nd4j.randn(DataType.FLOAT, 16, 16, 3, 48).mul(0.02);
        SDVariable weight = sd.constant("conv_weight", weightArr);
        INDArray biasArr = Nd4j.randn(DataType.FLOAT, 48).mul(0.01);
        SDVariable bias = sd.constant("conv_bias", biasArr);

        SDVariable conv = sd.cnn().conv2d("conv_out", input, weight, bias,
                org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.builder()
                        .kH(16).kW(16).sH(16).sW(16).pH(0).pW(0).dH(1).dW(1)
                        .dataFormat("NCHW").build());

        sd.setOutputs(Collections.singletonList("conv_out"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 3, 64, 64);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "conv_out");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros, max=" + refMax);
        log.info("Conv2d ref: shape={}, max={}", Arrays.toString(refOutput.shape()), refMax);

        // Apply FP16 quantization — this converts conv_weight from FLOAT to HALF
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("conv_out"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "conv_out");

        assertNotNull(optOutput, "FP16 conv2d output is null");
        assertFalse(optOutput.isNaN().any(), "FP16 conv2d output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "FP16 conv2d output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch after FP16");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("FP16 conv2d maxDiff={}", maxDiff);
        // FP16 has reduced precision but should be close
        assertTrue(maxDiff < 1.0, "FP16 conv2d diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // FP16 quantization: matmul weight quantization
    // Attention QKV projections and MLP layers use matmul with large FP32 weights.
    // FP16 pass converts these to HALF. The matmul must handle HALF*FLOAT correctly.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFP16_MatmulWeightQuantization() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 128;
        int mlpDim = hiddenDim * 4;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);
        // Large 2D weight — FP16 target
        INDArray w1Arr = Nd4j.randn(DataType.FLOAT, hiddenDim, mlpDim).mul(0.02);
        SDVariable w1 = sd.constant("mlp_w1", w1Arr);
        INDArray w2Arr = Nd4j.randn(DataType.FLOAT, mlpDim, hiddenDim).mul(0.02);
        SDVariable w2 = sd.constant("mlp_w2", w2Arr);

        SDVariable up = sd.mmul("mlp_up", input, w1);
        SDVariable act = sd.nn.gelu("mlp_gelu", up);
        SDVariable down = sd.mmul("output", act, w2);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "FP16 matmul output is null");
        assertFalse(optOutput.isNaN().any(), "FP16 matmul output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "FP16 matmul output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch after FP16 matmul");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("FP16 matmul maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1.0, "FP16 matmul diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // FP16 quantization: layernorm with mixed precision
    // Vision encoder has LayerNorm after embeddings with 1D gamma/beta (skipped
    // by FP16) and 2D+ projection weights (converted to HALF). The layernorm
    // itself stays FP32, but downstream matmul gets HALF weights. This tests
    // the boundary where FP32 activations meet HALF weights.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFP16_LayerNormThenMatmul() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 128;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // LayerNorm: gamma/beta are 1D (< 1024 elements) → stay FP32
        SDVariable gamma = sd.constant("ln_gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable beta = sd.constant("ln_beta", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable mean = input.mean(true, 2);
        SDVariable centered = input.sub(mean);
        SDVariable var = centered.mul(centered).mean(true, 2);
        SDVariable std = sd.math.sqrt(var.add(1e-5));
        SDVariable normed = centered.div(std);
        SDVariable lnOut = normed.mul(gamma).add(beta);

        // Projection: large 2D weight → converted to HALF
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable w = sd.constant("proj_weight", wArr);
        SDVariable output = sd.mmul("output", lnOut, w);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "FP16 LN+matmul output is null");
        assertFalse(optOutput.isNaN().any(), "FP16 LN+matmul output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "FP16 LN+matmul output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("FP16 LN+matmul maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1.0, "FP16 LN+matmul diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Cast chain removal: BOOL → LONG → FLOAT
    // Vision encoder has attention masks (BOOL) cast through LONG to FLOAT.
    // RemoveRedundantCasts may try to fuse or remove casts in this chain.
    // If it incorrectly removes a cast, the mask dtype is wrong and attention
    // produces garbage.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testCastChainRemoval_BoolToFloat() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);
        SDVariable mask = sd.placeHolder("mask", DataType.BOOL, 1, seqLen);

        // Cast chain: BOOL → LONG → FLOAT (mimics vision encoder mask pipeline)
        SDVariable maskLong = mask.castTo("mask_long", DataType.INT64);
        SDVariable maskFloat = maskLong.castTo("mask_float", DataType.FLOAT);

        // Use mask to zero out certain positions (like attention masking)
        SDVariable maskExpanded = sd.expandDims("mask_expand", maskFloat, 2);
        SDVariable output = input.mul("output", maskExpanded);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        // Mask with some zeros — positions 0,5,10 are masked out
        INDArray maskArr = Nd4j.ones(DataType.BOOL, 1, seqLen);
        maskArr.putScalar(0, 0, 0);
        maskArr.putScalar(0, 5, 0);
        maskArr.putScalar(0, 10, 0);

        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr, "mask", maskArr), "output");

        assertNotNull(refOutput);
        // Verify masking worked — positions 0,5,10 should be zero
        double pos0Max = refOutput.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).amaxNumber().doubleValue();
        assertEquals(0.0, pos0Max, 1e-10, "Masked position 0 should be zero");
        double pos1Max = refOutput.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).amaxNumber().doubleValue();
        assertTrue(pos1Max > 1e-6, "Unmasked position 1 should be non-zero");

        // Apply quantization optimizer (which includes RemoveRedundantCasts)
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr, "mask", maskArr), "output");

        assertNotNull(optOutput, "Cast chain output is null");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch after cast optimization");

        // Assert masking still works correctly
        double optPos0Max = optOutput.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).amaxNumber().doubleValue();
        assertEquals(0.0, optPos0Max, 1e-10, "Optimized: masked position 0 should still be zero");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Cast chain removal maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Cast chain removal corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Constant folding with LONG indices
    // Vision encoder uses LONG constants for position indices fed to Gather ops.
    // ConstantFunctionOptimizations may fold computations involving these constants.
    // If folding changes the dtype or values, position embeddings break.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testConstantFolding_LongPositionIndices() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int vocabSize = 64;
        int hiddenDim = 32;
        int seqLen = 16;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Position embedding table (VARIABLE, large 2D)
        INDArray embedTable = Nd4j.randn(DataType.FLOAT, vocabSize, hiddenDim).mul(0.02);
        SDVariable posEmbeddings = sd.constant("pos_embed_table", embedTable);

        // LONG position indices (constant) — this is what the vision encoder does
        INDArray posIndices = Nd4j.arange(0, seqLen).castTo(DataType.INT64).reshape(seqLen);
        SDVariable positions = sd.constant("positions", posIndices);

        // Gather position embeddings using LONG indices
        SDVariable posEmbed = sd.gather("pos_gathered", posEmbeddings, positions, 0);
        SDVariable posExpanded = sd.expandDims("pos_expand", posEmbed, 0);

        // Add position embeddings to input
        SDVariable output = input.add("output", posExpanded);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        // Apply constant folding
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ConstantFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Constant folding output is null");
        assertFalse(optOutput.isNaN().any(), "Constant folding output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Constant folding LONG indices maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Constant folding corrupted LONG indices: maxDiff=" + maxDiff);

        // Verify the position constant values survived in the optimized graph
        INDArray optPositions = optimized.getConstantArrays().getArray("positions");
        if (optPositions != null) {
            assertEquals(DataType.INT64, optPositions.dataType(),
                    "Position indices dtype changed from INT64 to " + optPositions.dataType());
            for (int i = 0; i < seqLen; i++) {
                assertEquals(i, optPositions.getLong(i),
                        "Position index[" + i + "] changed from " + i + " to " + optPositions.getLong(i));
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Identity/reshape removal in embedding pipeline
    // Vision encoder does: conv2d → reshape [1,768,1024] → transpose [1,1024,768]
    // IdentityFunctionOptimizations removes identity reshapes. If it incorrectly
    // removes a reshape that actually changes layout, shapes break downstream.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testIdentityRemoval_ReshapeTransposeChain() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int channels = 48;
        int h = 4, w = 4;
        int seqLen = h * w; // 16

        // Simulate post-conv output: [1, 48, 4, 4]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, channels, h, w);

        // Reshape to [1, 48, 16] — flattens spatial dims
        SDVariable reshaped = sd.reshape("reshape1", input, batch, channels, seqLen);

        // Transpose to [1, 16, 48] — sequence-first for transformer
        SDVariable transposed = sd.permute("transpose1", reshaped, 0, 2, 1);

        // Identity reshape (same shape) — should be removable
        SDVariable identityReshape = sd.reshape("identity_reshape", transposed, batch, seqLen, channels);

        // Linear projection
        INDArray projWArr = Nd4j.randn(DataType.FLOAT, channels, channels).mul(0.02);
        SDVariable projW = sd.constant("proj_w", projWArr);
        SDVariable output = sd.mmul("output", identityReshape, projW);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, channels, h, w);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new IdentityFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Identity removal output is null");
        assertFalse(optOutput.isNaN().any(), "Identity removal output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Identity removal maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Identity removal corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Algebraic simplification near zero/one constants
    // Vision encoder has add-zero and mul-one patterns from ONNX import artifacts.
    // AlgebraicOptimizations removes these. If it misidentifies a non-trivial op
    // as algebraic identity, computation is lost.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testAlgebraicSimplification_AddZeroMulOne() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Add zero (should be removable)
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable addZero = input.add("add_zero", zero);

        // Mul one (should be removable)
        SDVariable one = sd.constant("one", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable mulOne = addZero.mul("mul_one", one);

        // Real computation after simplifiable ops
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable w = sd.constant("proj_w", wArr);
        SDVariable output = sd.mmul("output", mulOne, w);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new AlgebraicOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Algebraic simplification output is null");
        assertFalse(optOutput.isNaN().any(), "Algebraic simplification output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Algebraic simplification output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Algebraic simplification maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Algebraic simplification corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // DCE (dead code elimination) with branching graph
    // Vision encoder has branches (attention mask computation, position ID
    // computation) that feed into the main path. DCE walks backward from outputs
    // and removes unreachable ops. If the backward walk misses a branch, live
    // ops get removed and the model produces wrong/zero output.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testDCE_PreservesBranchingGraph() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Branch A: layernorm path
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable mean = input.mean(true, 2);
        SDVariable centered = input.sub(mean);
        SDVariable var = centered.mul(centered).mean(true, 2);
        SDVariable std = sd.math.sqrt(var.add(1e-5));
        SDVariable normed = centered.div(std).mul(gamma);

        // Branch B: residual bypass
        SDVariable scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 1).mul(0.5));
        SDVariable scaled = input.mul(scale);

        // Branch C: dead branch (should be eliminated, but must not affect A or B)
        INDArray deadW = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim);
        SDVariable deadWeight = sd.constant("dead_weight", deadW);
        SDVariable deadBranch = sd.mmul("dead_out", input, deadWeight);
        // dead_out is never used by the output — DCE should remove it

        // Merge A + B
        SDVariable output = normed.add("output", scaled);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Full optimizer includes DCE as first pass (UnusedFunctionOptimizations)
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new UnusedFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "DCE output is null");
        assertFalse(optOutput.isNaN().any(), "DCE output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "DCE output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("DCE maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "DCE corrupted output: maxDiff=" + maxDiff);

        // Verify the dead branch was actually removed
        assertFalse(optimized.getOps().containsKey("dead_out"),
                "DCE should have removed dead_out op");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Activation fusion: gelu pattern
    // Vision encoder MLP uses GELU activation. ActivationFusionOptimizations
    // may try to fuse sigmoid*x→swish or detect SwiGLU. Must not incorrectly
    // fuse GELU into something else.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testActivationFusion_GeluPreservation() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;
        int mlpDim = hiddenDim * 4;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        INDArray w1Arr = Nd4j.randn(DataType.FLOAT, hiddenDim, mlpDim).mul(0.02);
        SDVariable w1 = sd.constant("w1", w1Arr);
        SDVariable up = sd.mmul("up", input, w1);
        SDVariable act = sd.nn.gelu("gelu", up);

        INDArray w2Arr = Nd4j.randn(DataType.FLOAT, mlpDim, hiddenDim).mul(0.02);
        SDVariable w2 = sd.constant("w2", w2Arr);
        SDVariable output = sd.mmul("output", act, w2);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ActivationFusionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Activation fusion output is null");
        assertFalse(optOutput.isNaN().any(), "Activation fusion output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Activation fusion output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Activation fusion maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Activation fusion corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Normalization fusion: layernorm-like pattern
    // Vision encoder has LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta.
    // NormalizationFusionOptimizations may fuse this into a single op.
    // The fused op must produce identical values.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testNormalizationFusion_LayerNormPattern() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 128;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Manual LayerNorm pattern (exactly as ONNX import produces)
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, hiddenDim).mul(1.1));
        SDVariable beta = sd.constant("beta", Nd4j.randn(DataType.FLOAT, hiddenDim).mul(0.01));
        SDVariable mean = input.mean(true, 2);
        SDVariable centered = input.sub(mean);
        SDVariable var = centered.mul(centered).mean(true, 2);
        SDVariable std = sd.math.sqrt(var.add(1e-5));
        SDVariable normed = centered.div(std);
        SDVariable output = normed.mul(gamma).add("output", beta);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        // Verify layernorm properties: output should have ~zero mean and ~1.1 std per position
        INDArray outputMean = refOutput.mean(2);
        double meanOfMeans = Nd4j.math().abs(outputMean).meanNumber().doubleValue();
        assertTrue(meanOfMeans < 0.5, "LayerNorm output mean should be near zero, got " + meanOfMeans);

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new NormalizationFusionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Norm fusion output is null");
        assertFalse(optOutput.isNaN().any(), "Norm fusion output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Normalization fusion maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "Normalization fusion corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Full pipeline: conv2d → reshape → transpose → layernorm → matmul
    // Combines all patterns from the vision encoder embedding pipeline.
    // Tests all passes together to catch interactions between optimizers.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFullPipeline_EmbeddingBlock() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int inChannels = 3;
        int outChannels = 48;
        int imgSize = 64;
        int patchSize = 16;
        int numPatches = (imgSize / patchSize) * (imgSize / patchSize); // 16

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, inChannels, imgSize, imgSize);

        // Patch embedding conv2d — SameDiff conv2d expects [kH, kW, inC, outC] (YXIO)
        INDArray convW = Nd4j.randn(DataType.FLOAT, patchSize, patchSize, inChannels, outChannels).mul(0.02);
        SDVariable weight = sd.constant("conv_w", convW);
        INDArray convB = Nd4j.randn(DataType.FLOAT, outChannels).mul(0.01);
        SDVariable bias = sd.constant("conv_b", convB);
        SDVariable conv = sd.cnn().conv2d("conv", input, weight, bias,
                org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.builder()
                        .kH(patchSize).kW(patchSize).sH(patchSize).sW(patchSize)
                        .pH(0).pW(0).dH(1).dW(1).dataFormat("NCHW").build());

        // Reshape: [1, 48, 4, 4] → [1, 48, 16]
        SDVariable reshaped = sd.reshape("reshape", conv, batch, outChannels, numPatches);

        // Transpose: [1, 48, 16] → [1, 16, 48]
        SDVariable transposed = sd.permute("transpose", reshaped, 0, 2, 1);

        // LayerNorm
        SDVariable gamma = sd.constant("ln_gamma", Nd4j.ones(DataType.FLOAT, outChannels));
        SDVariable beta = sd.constant("ln_beta", Nd4j.zeros(DataType.FLOAT, outChannels));
        SDVariable mean = transposed.mean(true, 2);
        SDVariable centered = transposed.sub(mean);
        SDVariable var = centered.mul(centered).mean(true, 2);
        SDVariable std = sd.math.sqrt(var.add(1e-5));
        SDVariable normed = centered.div(std);
        SDVariable lnOut = normed.mul(gamma).add(beta);

        // Linear projection
        INDArray projW = Nd4j.randn(DataType.FLOAT, outChannels, outChannels).mul(0.02);
        SDVariable projWeight = sd.constant("proj_w", projW);
        SDVariable output = sd.mmul("output", lnOut, projWeight);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, inChannels, imgSize, imgSize);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");
        log.info("Full pipeline ref: shape={}, max={}", Arrays.toString(refOutput.shape()), refMax);

        // Test with full optimizer, FP16 OFF — structural optimizations only
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff optNoFP16 = GraphOptimizer.optimize(sd, Collections.singletonList("output"));
        INDArray optNoFP16Out = optNoFP16.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optNoFP16Out, "Full optimizer (no FP16) output is null");
        assertFalse(optNoFP16Out.isNaN().any(), "Full optimizer (no FP16) output has NaN");
        double noFP16Max = optNoFP16Out.amaxNumber().doubleValue();
        assertTrue(noFP16Max > 1e-6, "Full optimizer (no FP16) output is all zeros, max=" + noFP16Max);
        assertArrayEquals(refOutput.shape(), optNoFP16Out.shape(), "Shape mismatch (no FP16)");

        double noFP16Diff = refOutput.sub(optNoFP16Out).amaxNumber().doubleValue();
        log.info("Full pipeline (no FP16) maxDiff={}", noFP16Diff);
        assertTrue(noFP16Diff < 1e-3, "Full optimizer (no FP16) corrupted output: maxDiff=" + noFP16Diff);

        // Test with full optimizer, FP16 ON — the production config
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optFP16 = GraphOptimizer.optimize(sd, Collections.singletonList("output"));
        INDArray optFP16Out = optFP16.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optFP16Out, "Full optimizer (FP16) output is null");
        assertFalse(optFP16Out.isNaN().any(), "Full optimizer (FP16) output has NaN");
        double fp16Max = optFP16Out.amaxNumber().doubleValue();
        assertTrue(fp16Max > 1e-6, "Full optimizer (FP16) output is all zeros, max=" + fp16Max);
        assertArrayEquals(refOutput.shape(), optFP16Out.shape(), "Shape mismatch (FP16)");

        double fp16Diff = refOutput.sub(optFP16Out).amaxNumber().doubleValue();
        log.info("Full pipeline (FP16) maxDiff={}", fp16Diff);
        assertTrue(fp16Diff < 1.0, "Full optimizer (FP16) corrupted output: maxDiff=" + fp16Diff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // dup() round-trip: vision-encoder-like graph
    // GraphOptimizer calls sd.dup() before applying passes. If dup() itself
    // corrupts constants (e.g., LONG endianness, FP32 precision loss during
    // SDNB serialization), all optimizer results are wrong.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testDupRoundtrip_MixedDtypeConstants() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // FLOAT constant (weight matrix)
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable w = sd.constant("weight", wArr);

        // LONG constant (position indices)
        INDArray posArr = Nd4j.arange(0, seqLen).castTo(DataType.INT64);
        SDVariable positions = sd.constant("positions", posArr);

        // BOOL constant (mask)
        INDArray maskArr = Nd4j.ones(DataType.BOOL, seqLen);
        maskArr.putScalar(0, 0);
        maskArr.putScalar(seqLen - 1, 0);
        SDVariable mask = sd.constant("mask", maskArr);

        // Use all three dtypes in computation
        SDVariable matmul = sd.mmul("matmul", input, w);
        SDVariable posFloat = positions.castTo("pos_float", DataType.FLOAT);
        SDVariable posExpanded = sd.expandDims("pos_expand", posFloat, 0);
        SDVariable posExpanded2 = sd.expandDims("pos_expand2", posExpanded, 2);
        SDVariable withPos = matmul.add(posExpanded2);

        SDVariable maskFloat = mask.castTo("mask_float", DataType.FLOAT);
        SDVariable maskExpanded = sd.expandDims("mask_expand", maskFloat, 0);
        SDVariable maskExpanded2 = sd.expandDims("mask_expand2", maskExpanded, 2);
        SDVariable output = withPos.mul("output", maskExpanded2);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        // Verify mask worked — first and last positions should be zero
        double pos0Max = refOutput.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).amaxNumber().doubleValue();
        assertEquals(0.0, pos0Max, 1e-10, "Position 0 should be masked to zero");

        // dup() round-trip
        SameDiff duped = sd.dup();
        if (duped.outputs() == null || duped.outputs().isEmpty()) {
            duped.setOutputs(new ArrayList<>(sd.outputs()));
        }

        INDArray dupedOutput = duped.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(dupedOutput, "dup() output is null");
        assertFalse(dupedOutput.isNaN().any(), "dup() output has NaN");
        assertArrayEquals(refOutput.shape(), dupedOutput.shape(), "dup() shape mismatch");

        double maxDiff = refOutput.sub(dupedOutput).amaxNumber().doubleValue();
        log.info("dup() mixed-dtype maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "dup() corrupted mixed-dtype graph: maxDiff=" + maxDiff);

        // Verify LONG constant survived with correct values
        INDArray dupedPos = duped.getConstantArrays().getArray("positions");
        assertNotNull(dupedPos, "LONG positions constant missing after dup()");
        assertEquals(DataType.INT64, dupedPos.dataType(),
                "LONG positions dtype changed to " + dupedPos.dataType());
        for (int i = 0; i < seqLen; i++) {
            assertEquals(i, dupedPos.getLong(i),
                    "Position[" + i + "] wrong after dup(): expected " + i + " got " + dupedPos.getLong(i));
        }

        // Verify BOOL constant survived
        INDArray dupedMask = duped.getConstantArrays().getArray("mask");
        assertNotNull(dupedMask, "BOOL mask constant missing after dup()");
        assertEquals(0, dupedMask.getInt(0), "Mask[0] should be 0 after dup()");
        assertEquals(1, dupedMask.getInt(1), "Mask[1] should be 1 after dup()");
        assertEquals(0, dupedMask.getInt(seqLen - 1), "Mask[last] should be 0 after dup()");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // FP16 quantization: attention-like QKV projection
    // Vision encoder attention: Q,K,V are projected from hidden dim, then
    // Q*K^T/sqrt(d) + mask → softmax → *V. If QKV weights are HALF but
    // activations are FLOAT, the matmul must handle mixed types. If not,
    // attention scores are zero → softmax outputs uniform → output is garbage.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFP16_AttentionQKVProjection() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seqLen = 16;
        int hiddenDim = 64;
        int headDim = 16;
        int numHeads = hiddenDim / headDim; // 4

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenDim);

        // Q, K, V projection weights (large 2D → FP16 targets)
        INDArray wqArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wkArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wvArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wq = sd.constant("wq", wqArr);
        SDVariable wk = sd.constant("wk", wkArr);
        SDVariable wv = sd.constant("wv", wvArr);

        // Project Q, K, V
        SDVariable q = sd.mmul("q_proj", input, wq);
        SDVariable k = sd.mmul("k_proj", input, wk);
        SDVariable v = sd.mmul("v_proj", input, wv);

        // Reshape to multi-head: [1, 16, 64] → [1, 16, 4, 16] → [1, 4, 16, 16]
        SDVariable qReshaped = sd.reshape("q_reshape", q, batch, seqLen, numHeads, headDim);
        SDVariable qTransposed = sd.permute("q_transpose", qReshaped, 0, 2, 1, 3);
        SDVariable kReshaped = sd.reshape("k_reshape", k, batch, seqLen, numHeads, headDim);
        SDVariable kTransposed = sd.permute("k_transpose", kReshaped, 0, 2, 3, 1); // [1,4,16,16] for K^T
        SDVariable vReshaped = sd.reshape("v_reshape", v, batch, seqLen, numHeads, headDim);
        SDVariable vTransposed = sd.permute("v_transpose", vReshaped, 0, 2, 1, 3);

        // Attention: Q * K^T / sqrt(d)
        SDVariable scores = sd.mmul("attn_scores", qTransposed, kTransposed);
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0 / Math.sqrt(headDim)));
        SDVariable scaledScores = scores.mul("scaled_scores", scale);
        SDVariable attnWeights = sd.nn.softmax("softmax", scaledScores, 3);

        // Attention output: weights * V
        SDVariable attnOut = sd.mmul("attn_out", attnWeights, vTransposed);

        // Reshape back: [1, 4, 16, 16] → [1, 16, 64]
        SDVariable attnPermuted = sd.permute("attn_permute", attnOut, 0, 2, 1, 3);
        SDVariable output = sd.reshape("output", attnPermuted, batch, seqLen, hiddenDim);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Verify attention output is not uniform (softmax didn't get garbage input)
        double refStd = refOutput.stdNumber().doubleValue();
        assertTrue(refStd > 1e-4, "Reference output std is too low (" + refStd + "), attention may be broken");

        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "FP16 attention output is null");
        assertFalse(optOutput.isNaN().any(), "FP16 attention output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "FP16 attention output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        // Verify attention output still has meaningful variance
        double optStd = optOutput.stdNumber().doubleValue();
        assertTrue(optStd > 1e-4, "FP16 attention output std is too low (" + optStd + "), attention likely broken");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("FP16 attention maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1.0, "FP16 attention diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Attention fusion: full self-attention → DotProductAttentionV2
    // The AttentionFusionOptimizations pass detects the pattern:
    //   Q@K^T → scale → [mask add] → softmax → @V
    // and replaces it with a fused DotProductAttentionV2 op.
    // If the fusion incorrectly wires inputs (e.g., swaps Q/K/V, loses scale,
    // drops mask), the fused op produces wrong values.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testAttentionFusion_SelfAttentionPattern() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seqLen = 8;
        int hiddenDim = 64;
        int headDim = 16;
        int numHeads = hiddenDim / headDim; // 4

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenDim);

        // QKV projections
        INDArray wqArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wkArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wvArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wq = sd.constant("wq", wqArr);
        SDVariable wk = sd.constant("wk", wkArr);
        SDVariable wv = sd.constant("wv", wvArr);

        SDVariable q = sd.mmul("q_proj", input, wq);
        SDVariable k = sd.mmul("k_proj", input, wk);
        SDVariable v = sd.mmul("v_proj", input, wv);

        // Reshape to multi-head and transpose to [B, numHeads, seqLen, headDim]
        SDVariable qr = sd.reshape("qr", q, batch, seqLen, numHeads, headDim);
        SDVariable qt = sd.permute("qt", qr, 0, 2, 1, 3);
        SDVariable kr = sd.reshape("kr", k, batch, seqLen, numHeads, headDim);
        SDVariable kt = sd.permute("kt", kr, 0, 2, 3, 1); // transposed for Q@K^T
        SDVariable vr = sd.reshape("vr", v, batch, seqLen, numHeads, headDim);
        SDVariable vt = sd.permute("vt", vr, 0, 2, 1, 3);

        // Q@K^T / sqrt(d)
        SDVariable scores = sd.mmul("qk", qt, kt);
        SDVariable scaled = scores.div("scaled", sd.constant("sqrt_d",
                Nd4j.scalar(DataType.FLOAT, Math.sqrt(headDim))));

        // Softmax
        SDVariable weights = sd.nn.softmax("attn_softmax", scaled, 3);

        // Weighted sum: softmax @ V
        SDVariable attnOut = sd.mmul("attn_v", weights, vt);

        // Transpose back and reshape: [B, numHeads, seqLen, headDim] → [B, seqLen, hiddenDim]
        SDVariable outPermuted = sd.permute("out_perm", attnOut, 0, 2, 1, 3);
        SDVariable output = sd.reshape("output", outPermuted, batch, seqLen, hiddenDim);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");
        double refStd = refOutput.stdNumber().doubleValue();
        assertTrue(refStd > 1e-4, "Reference std too low: " + refStd);

        // Apply attention fusion pass only
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new AttentionFusionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Attention fusion output is null");
        assertFalse(optOutput.isNaN().any(), "Attention fusion output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Attention fusion output is all zeros, max=" + optMax);
        double optStd = optOutput.stdNumber().doubleValue();
        assertTrue(optStd > 1e-4, "Attention fusion std too low: " + optStd);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double attnDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Attention fusion maxDiff={}", attnDiff);
        assertTrue(attnDiff < 1e-3, "Attention fusion corrupted output: maxDiff=" + attnDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Linear fusion: matmul + add → XwPlusB
    // LinearFusionOptimizations detects matmul followed by bias add and fuses
    // into a single XwPlusB op. If fusion drops the bias or transposes wrong,
    // the output is wrong. Tests both standard Mmul+Add and the TensorMmul path.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testLinearFusion_MatmulPlusBias() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int inDim = 64;
        int outDim = 128;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, inDim);

        // Matmul + bias (the pattern LinearFusion targets)
        INDArray wArr = Nd4j.randn(DataType.FLOAT, inDim, outDim).mul(0.02);
        SDVariable w = sd.constant("weight", wArr);
        SDVariable mm = sd.mmul("matmul", input, w);

        INDArray biasArr = Nd4j.randn(DataType.FLOAT, outDim).mul(0.01);
        SDVariable bias = sd.constant("bias", biasArr);
        SDVariable withBias = mm.add("add_bias", bias);

        // Another matmul+bias layer to test chained fusion
        INDArray w2Arr = Nd4j.randn(DataType.FLOAT, outDim, inDim).mul(0.02);
        SDVariable w2 = sd.constant("weight2", w2Arr);
        SDVariable mm2 = sd.mmul("matmul2", withBias, w2);

        INDArray bias2Arr = Nd4j.randn(DataType.FLOAT, inDim).mul(0.01);
        SDVariable bias2 = sd.constant("bias2", bias2Arr);
        SDVariable output = mm2.add("output", bias2);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, inDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Apply linear fusion
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new LinearFusionOptimizations()));

        // XwPlusB (the fused op) is not supported by DSP native plan serialization.
        // Disable DSP so the standard execution path is used for validation.
        optimized.setDspAutoCompileEnabled(false);
        optimized.setDspNativeAutoCompileEnabled(false);

        // After fusion the graph output name may have changed to the fused op's output
        String optOutName = optimized.outputs().get(0);
        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), optOutName);

        assertNotNull(optOutput, "Linear fusion output is null (output var: " + optOutName + ")");
        assertFalse(optOutput.isNaN().any(), "Linear fusion output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Linear fusion output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double linDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Linear fusion maxDiff={}", linDiff);
        assertTrue(linDiff < 1e-4, "Linear fusion corrupted output: maxDiff=" + linDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Shape fusion: chained permutes and reshapes
    // ShapeFunctionOptimizations fuses consecutive permutes into one composed
    // permutation and consecutive reshapes into one. If the composition is
    // wrong, the tensor layout is silently corrupted → garbage downstream.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testShapeFusion_ChainedPermutes() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seqLen = 8;
        int numHeads = 4;
        int headDim = 16;

        // [B, S, H, D]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Chain of permutes: [B,S,H,D] → [B,H,S,D] → [B,H,D,S] → [B,S,D,H]
        SDVariable p1 = sd.permute("perm1", input, 0, 2, 1, 3);  // B,H,S,D
        SDVariable p2 = sd.permute("perm2", p1, 0, 1, 3, 2);      // B,H,D,S
        SDVariable p3 = sd.permute("perm3", p2, 0, 2, 3, 1);      // B,D,S,H

        // Flatten last two dims and project
        SDVariable flat = sd.reshape("flat", p3, batch, headDim, seqLen * numHeads);
        INDArray wArr = Nd4j.randn(DataType.FLOAT, seqLen * numHeads, seqLen * numHeads).mul(0.02);
        SDVariable w = sd.constant("proj_w", wArr);
        SDVariable output = sd.mmul("output", flat, w);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ShapeFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Shape fusion output is null");
        assertFalse(optOutput.isNaN().any(), "Shape fusion output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double shapeDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Shape fusion (permutes) maxDiff={}", shapeDiff);
        assertTrue(shapeDiff < 1e-5, "Shape fusion corrupted permutes: maxDiff=" + shapeDiff);
    }

    @Test
    public void testShapeFusion_ChainedReshapes() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int total = 256;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, total);

        // Chain of reshapes: [1,256] → [1,16,16] → [1,4,4,16] → [1,256]
        SDVariable r1 = sd.reshape("r1", input, 1, 16, 16);
        SDVariable r2 = sd.reshape("r2", r1, 1, 4, 4, 16);
        SDVariable r3 = sd.reshape("r3", r2, 1, total);

        // Do something with the final shape
        INDArray wArr = Nd4j.randn(DataType.FLOAT, total, 64).mul(0.02);
        SDVariable w = sd.constant("w", wArr);
        SDVariable output = sd.mmul("output", r3, w);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, total);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ShapeFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Reshape fusion output is null");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double reshapeDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Shape fusion (reshapes) maxDiff={}", reshapeDiff);
        assertTrue(reshapeDiff < 1e-5, "Shape fusion corrupted reshapes: maxDiff=" + reshapeDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CuDNN conv2d NCHW→NHWC conversion
    // CuDNNFunctionOptimizations inserts permute ops before/after conv2d to
    // convert NCHW activations to NHWC for tensor core acceleration, and
    // permutes weights to OYXI format. If the permutations are wrong, the
    // conv produces garbage but downstream ops don't know.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testCuDNN_Conv2dLayoutConversion() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        // [kH, kW, inC, outC] YXIO format (SameDiff default)
        INDArray weightArr = Nd4j.randn(DataType.FLOAT, 4, 4, 3, 16).mul(0.02);
        SDVariable weight = sd.constant("conv_w", weightArr);
        INDArray biasArr = Nd4j.randn(DataType.FLOAT, 16).mul(0.01);
        SDVariable bias = sd.constant("conv_b", biasArr);

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, 32, 32);
        SDVariable conv = sd.cnn().conv2d("conv", input, weight, bias,
                org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig.builder()
                        .kH(4).kW(4).sH(4).sW(4).pH(0).pW(0).dH(1).dW(1)
                        .dataFormat("NCHW").build());

        // Feed conv output into a matmul to verify downstream is correct
        SDVariable flat = sd.reshape("flat", conv, 1, 16 * 8 * 8);
        INDArray wArr = Nd4j.randn(DataType.FLOAT, 16 * 8 * 8, 32).mul(0.02);
        SDVariable w = sd.constant("fc_w", wArr);
        SDVariable output = sd.mmul("output", flat, w);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 3, 32, 32);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new CuDNNFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "CuDNN conv output is null");
        assertFalse(optOutput.isNaN().any(), "CuDNN conv output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "CuDNN conv output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double cudnnDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("CuDNN conv layout conversion maxDiff={}", cudnnDiff);
        assertTrue(cudnnDiff < 1e-3, "CuDNN conv layout conversion corrupted output: maxDiff=" + cudnnDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // All passes combined on a multi-layer transformer block
    // Tests interactions between passes: attention fusion + linear fusion +
    // shape fusion + FP16 quantization + DCE all running together on a graph
    // with attention, MLP, layernorm, and residual connections.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testAllPasses_TransformerBlock() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seqLen = 8;
        int hiddenDim = 64;
        int headDim = 16;
        int numHeads = hiddenDim / headDim;
        int mlpDim = hiddenDim * 4;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenDim);

        // ─── LayerNorm 1 ───
        SDVariable gamma1 = sd.constant("ln1_gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable beta1 = sd.constant("ln1_beta", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable mean1 = input.mean(true, 2);
        SDVariable centered1 = input.sub(mean1);
        SDVariable var1 = centered1.mul(centered1).mean(true, 2);
        SDVariable std1 = sd.math.sqrt(var1.add(1e-5));
        SDVariable ln1 = centered1.div(std1).mul(gamma1).add(beta1);

        // ─── Self-Attention ───
        INDArray wqArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wkArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wvArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wq = sd.constant("attn_wq", wqArr);
        SDVariable wk = sd.constant("attn_wk", wkArr);
        SDVariable wv = sd.constant("attn_wv", wvArr);

        SDVariable q = sd.mmul("q", ln1, wq);
        SDVariable k = sd.mmul("k", ln1, wk);
        SDVariable v = sd.mmul("v", ln1, wv);

        // Multi-head reshape + permute
        SDVariable qr = sd.reshape("qr", q, batch, seqLen, numHeads, headDim);
        SDVariable qt = sd.permute("qt", qr, 0, 2, 1, 3);
        SDVariable kr = sd.reshape("kr", k, batch, seqLen, numHeads, headDim);
        SDVariable ktT = sd.permute("ktT", kr, 0, 2, 3, 1);
        SDVariable vr = sd.reshape("vr", v, batch, seqLen, numHeads, headDim);
        SDVariable vt = sd.permute("vt", vr, 0, 2, 1, 3);

        SDVariable scores = sd.mmul("scores", qt, ktT);
        SDVariable scaled = scores.div("scaled_scores",
                sd.constant("scale", Nd4j.scalar(DataType.FLOAT, Math.sqrt(headDim))));
        SDVariable attnW = sd.nn.softmax("attn_w", scaled, 3);
        SDVariable attnOut = sd.mmul("attn_out", attnW, vt);

        SDVariable attnPerm = sd.permute("attn_perm", attnOut, 0, 2, 1, 3);
        SDVariable attnFlat = sd.reshape("attn_flat", attnPerm, batch, seqLen, hiddenDim);

        // Output projection (matmul + bias → LinearFusion target)
        INDArray woArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wo = sd.constant("attn_wo", woArr);
        SDVariable attnProj = sd.mmul("attn_proj", attnFlat, wo);
        INDArray attnBiasArr = Nd4j.randn(DataType.FLOAT, hiddenDim).mul(0.01);
        SDVariable attnBias = sd.constant("attn_bias", attnBiasArr);
        SDVariable attnWithBias = attnProj.add("attn_add_bias", attnBias);

        // Residual 1
        SDVariable res1 = input.add("residual1", attnWithBias);

        // ─── LayerNorm 2 ───
        SDVariable gamma2 = sd.constant("ln2_gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable beta2 = sd.constant("ln2_beta", Nd4j.zeros(DataType.FLOAT, hiddenDim));
        SDVariable mean2 = res1.mean(true, 2);
        SDVariable centered2 = res1.sub(mean2);
        SDVariable var2 = centered2.mul(centered2).mean(true, 2);
        SDVariable std2 = sd.math.sqrt(var2.add(1e-5));
        SDVariable ln2 = centered2.div(std2).mul(gamma2).add(beta2);

        // ─── MLP ───
        INDArray w1Arr = Nd4j.randn(DataType.FLOAT, hiddenDim, mlpDim).mul(0.02);
        SDVariable mlpW1 = sd.constant("mlp_w1", w1Arr);
        SDVariable mlpUp = sd.mmul("mlp_up", ln2, mlpW1);
        INDArray b1Arr = Nd4j.randn(DataType.FLOAT, mlpDim).mul(0.01);
        SDVariable mlpB1 = sd.constant("mlp_b1", b1Arr);
        SDVariable mlpUpBias = mlpUp.add("mlp_up_bias", mlpB1);
        SDVariable mlpAct = sd.nn.gelu("mlp_gelu", mlpUpBias);

        INDArray w2Arr = Nd4j.randn(DataType.FLOAT, mlpDim, hiddenDim).mul(0.02);
        SDVariable mlpW2 = sd.constant("mlp_w2", w2Arr);
        SDVariable mlpDown = sd.mmul("mlp_down", mlpAct, mlpW2);
        INDArray b2Arr = Nd4j.randn(DataType.FLOAT, hiddenDim).mul(0.01);
        SDVariable mlpB2 = sd.constant("mlp_b2", b2Arr);
        SDVariable mlpDownBias = mlpDown.add("mlp_down_bias", mlpB2);

        // Residual 2
        SDVariable output = res1.add("output", mlpDownBias);

        // Dead branch — should be removed by DCE
        INDArray deadArr = Nd4j.randn(DataType.FLOAT, hiddenDim, 32);
        SDVariable deadW = sd.constant("dead_w", deadArr);
        SDVariable dead = sd.mmul("dead_branch", ln1, deadW);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");
        double refStd = refOutput.stdNumber().doubleValue();
        assertTrue(refStd > 1e-4, "Reference std too low: " + refStd);

        // Full optimizer — all passes interact
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"));
        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Full transformer block output is null");
        assertFalse(optOutput.isNaN().any(), "Full transformer block output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Full transformer block output is all zeros, max=" + optMax);
        double optStd = optOutput.stdNumber().doubleValue();
        assertTrue(optStd > 1e-4, "Full transformer block std too low: " + optStd);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double fullDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Full transformer block (all passes, no FP16) maxDiff={}", fullDiff);
        assertTrue(fullDiff < 1e-3, "Full transformer block all-passes corrupted output: maxDiff=" + fullDiff);

        // Verify dead branch removed
        assertFalse(optimized.getOps().containsKey("dead_branch"),
                "DCE should have removed dead_branch");

        // Now test with FP16 ON
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff imported2 = sd; // reuse — optimizer dups internally
        SameDiff optFP16 = GraphOptimizer.optimize(imported2, Collections.singletonList("output"));
        INDArray fp16Output = optFP16.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(fp16Output, "FP16 transformer output is null");
        assertFalse(fp16Output.isNaN().any(), "FP16 transformer output has NaN");
        double fp16Max = fp16Output.amaxNumber().doubleValue();
        assertTrue(fp16Max > 1e-6, "FP16 transformer output is all zeros, max=" + fp16Max);
        double fp16Std = fp16Output.stdNumber().doubleValue();
        assertTrue(fp16Std > 1e-4, "FP16 transformer std too low: " + fp16Std);

        double fp16Diff = refOutput.sub(fp16Output).amaxNumber().doubleValue();
        log.info("Full transformer block (all passes, FP16 ON) maxDiff={}", fp16Diff);
        assertTrue(fp16Diff < 1.0, "FP16 transformer block all-passes corrupted output: maxDiff=" + fp16Diff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Output variable survival after optimization
    // GraphOptimizer has complex output preservation logic. If the output
    // variable name is removed or redirected by a pass (e.g., fusion replaces
    // the output node), the graph returns wrong/empty results.
    // Tests that the output variable survives various optimization scenarios.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testOutputVariableSurvival_AfterFusion() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // The output variable IS the result of a matmul+bias add
        // Linear fusion will try to replace this add with XwPlusB
        // The output name "final_output" must survive
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable w = sd.constant("w", wArr);
        SDVariable mm = sd.mmul("mm", input, w);

        INDArray biasArr = Nd4j.randn(DataType.FLOAT, hiddenDim).mul(0.01);
        SDVariable bias = sd.constant("bias", biasArr);
        SDVariable output = mm.add("final_output", bias);

        sd.setOutputs(Collections.singletonList("final_output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "final_output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("final_output"));
        // LinearFusion produces XwPlusB which DSP native plan can't serialize
        optimized.setDspAutoCompileEnabled(false);
        optimized.setDspNativeAutoCompileEnabled(false);

        // Verify an output variable exists (fusion may rename it to the fused op's output)
        assertNotNull(optimized.outputs(), "Optimized graph has null outputs");
        assertFalse(optimized.outputs().isEmpty(), "Optimized graph has empty outputs");
        log.info("Output variable after optimization: {}", optimized.outputs());

        String outName = optimized.outputs().get(0);
        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), outName);

        assertNotNull(optOutput, "Output is null after fusion");
        assertFalse(optOutput.isNaN().any(), "Output has NaN after fusion");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double outDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Output survival maxDiff={}", outDiff);
        assertTrue(outDiff < 1e-4, "Output variable survival: values corrupted, maxDiff=" + outDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Multi-pass interaction: constant folding then FP16 quantization
    // Constant folding folds all-constant ops into new large constant arrays.
    // After folding, the folded constants become new 2D arrays that the FP16
    // pass should then convert. This tests the interaction where folding
    // produces arrays that are later targeted by quantization.
    // This was the bug scenario: OptimizedGraphArrayHolder would crash on
    // setArray when a function-backed entry already existed (FIXED: removeArray
    // before setArray).
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testMultiPass_ConstantFoldingThenFP16() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int hiddenDim = 64;
        int seqLen = 16;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Two constants that are both constant — their product can be folded.
        // The product is a large 2D matrix [hiddenDim, hiddenDim] (>= 1024 elements).
        // After folding, it should be a constant that FP16 quantization then converts.
        INDArray scaleArr = Nd4j.ones(DataType.FLOAT, hiddenDim, hiddenDim);
        SDVariable scaleMat = sd.constant("scale_mat", scaleArr);
        INDArray wBaseArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wBase = sd.constant("w_base", wBaseArr);

        // All-constant mul: scale_mat * w_base → large 2D result, will be folded
        SDVariable foldedWeight = scaleMat.mul("folded_weight", wBase);

        // Use folded weight with dynamic input
        SDVariable output = sd.mmul("output", input, foldedWeight);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Apply constant folding first, then FP16 — the interaction that caused the crash
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Arrays.asList(new ConstantFunctionOptimizations(), new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Constant fold + FP16 output is null");
        assertFalse(optOutput.isNaN().any(), "Constant fold + FP16 output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Constant fold + FP16 output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch after fold+FP16");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Constant fold + FP16 maxDiff={}", maxDiff);
        // FP16 tolerance after folding — should still be close
        assertTrue(maxDiff < 1.0, "Constant fold + FP16 diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Redundant cast removal around attention masks: BOOL → LONG → FLOAT
    // RemoveRedundantCasts should fuse cast(BOOL→LONG) + cast(LONG→FLOAT) into
    // a single cast(BOOL→FLOAT). The mask must still zero out the right positions.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testRedundantCastRemoval_AttentionMaskBoolToLongToFloat() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int seqLen = 12;
        int hiddenDim = 32;

        // Attention scores (dynamic) and BOOL mask (dynamic placeholder)
        SDVariable scores = sd.placeHolder("scores", DataType.FLOAT, 1, seqLen, seqLen);
        SDVariable boolMask = sd.placeHolder("bool_mask", DataType.BOOL, 1, seqLen);

        // BOOL → LONG → FLOAT cast chain (the pattern from ONNX attention mask import)
        SDVariable maskLong = boolMask.castTo("mask_to_long", DataType.INT64);
        SDVariable maskFloat = maskLong.castTo("mask_to_float", DataType.FLOAT);

        // Expand mask to [1, 1, seqLen] then broadcast with scores [1, seqLen, seqLen]
        SDVariable maskExpanded = sd.expandDims("mask_expand_cast", maskFloat, 1);

        // Multiply scores by mask: positions where mask=0 → score goes to 0
        SDVariable maskedScores = scores.mul("masked_scores_cast", maskExpanded);

        // Softmax over masked scores
        SDVariable attnWeights = sd.nn.softmax("attn_weights_cast", maskedScores, -1);
        SDVariable output = sd.reshape("output", attnWeights, 1, seqLen, seqLen);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray scoresArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, seqLen).mul(0.1);
        // Mask: positions 0..8 valid (true), positions 9..11 masked (false)
        INDArray maskArr = Nd4j.ones(DataType.BOOL, 1, seqLen);
        maskArr.putScalar(new int[]{0, 9}, 0);
        maskArr.putScalar(new int[]{0, 10}, 0);
        maskArr.putScalar(new int[]{0, 11}, 0);

        INDArray refOutput = sd.outputSingle(Map.of("scores", scoresArr, "bool_mask", maskArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        // Softmax output rows should sum to ~1
        INDArray rowSums = refOutput.sum(true, -1);
        double maxRowSumDev = rowSums.sub(1.0).amaxNumber().doubleValue();
        assertTrue(maxRowSumDev < 1e-4, "Softmax rows don't sum to 1: max deviation=" + maxRowSumDev);

        // Apply QuantizationOptimizations which includes RemoveRedundantCasts
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("scores", scoresArr, "bool_mask", maskArr), "output");

        assertNotNull(optOutput, "Cast-optimized output is null");
        assertFalse(optOutput.isNaN().any(), "Cast-optimized output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch after cast removal");

        // Values must match — cast fusion must not change semantics
        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Redundant cast removal (attention mask) maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Cast removal changed attention mask behavior: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Graph with multiple outputs: all outputs must survive optimization
    // The output preservation logic in GraphOptimizer must keep ALL registered
    // output variables alive, not just the first one.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testMultipleOutputs_AllSurviveOptimization() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int seqLen = 8;
        int hiddenDim = 32;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Branch 1: layernorm-like normalization → output_a
        SDVariable gamma1 = sd.constant("mo_gamma1", Nd4j.ones(DataType.FLOAT, hiddenDim));
        SDVariable mean1 = input.mean(true, 2);
        SDVariable centered1 = input.sub(mean1);
        SDVariable var1 = centered1.mul(centered1).mean(true, 2);
        SDVariable std1 = sd.math.sqrt(var1.add(1e-5));
        SDVariable outputA = centered1.div(std1).mul("output_a", gamma1);

        // Branch 2: matmul projection → output_b
        INDArray wArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable w = sd.constant("mo_proj_w", wArr);
        SDVariable outputB = sd.mmul("output_b", input, w);

        // Branch 3: element-wise activation → output_c
        SDVariable outputC = sd.nn.gelu("output_c", input);

        // Dead branch: not connected to any output — DCE should remove it
        INDArray deadW = Nd4j.randn(DataType.FLOAT, hiddenDim, 16);
        SDVariable deadWeight = sd.constant("mo_dead_w", deadW);
        SDVariable dead = sd.mmul("mo_dead_op", input, deadWeight);

        // Register all 3 real outputs
        List<String> outputNames = Arrays.asList("output_a", "output_b", "output_c");
        sd.setOutputs(outputNames);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        Map<String, INDArray> refMap = sd.output(Map.of("input", inputArr), outputNames);

        for (String outName : outputNames) {
            assertNotNull(refMap.get(outName), "Reference output null: " + outName);
            assertFalse(refMap.get(outName).isNaN().any(), "Reference NaN: " + outName);
        }

        // Run full optimizer — all 3 outputs must survive
        SameDiff optimized = GraphOptimizer.optimize(sd, outputNames);
        // Disable DSP to isolate optimizer correctness from DSP execution issues
        optimized.setDspAutoCompileEnabled(false);
        optimized.setDspNativeAutoCompileEnabled(false);

        List<String> optOutputNames = optimized.outputs();
        assertNotNull(optOutputNames, "Optimized graph has null outputs");
        assertEquals(outputNames.size(), optOutputNames.size(),
                "Output count changed. Original: " + outputNames + ", Optimized: " + optOutputNames);
        log.info("Optimized output names: {}", optOutputNames);

        // Request outputs using the optimized graph's output names (may have been renamed by algebraic simplification)
        Map<String, INDArray> optMap = optimized.output(Map.of("input", inputArr), optOutputNames);

        // Compare by position — output at position i in optimized should match position i in reference
        for (int i = 0; i < outputNames.size(); i++) {
            String refName = outputNames.get(i);
            String optName = optOutputNames.get(i);
            INDArray ref = refMap.get(refName);
            INDArray opt = optMap.get(optName);
            assertNotNull(opt, "Optimized output null: " + optName + " (was " + refName + ")");
            assertFalse(opt.isNaN().any(), "Optimized output NaN: " + optName);
            assertArrayEquals(ref.shape(), opt.shape(), "Shape mismatch: " + optName + " (was " + refName + ")");

            double diff = ref.sub(opt).amaxNumber().doubleValue();
            log.info("Multi-output {} (was {}) maxDiff={}", optName, refName, diff);
            assertTrue(diff < 1e-4, "Multi-output corruption on " + optName + ": maxDiff=" + diff);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Constant folding that changes dtype: Cast(FLOAT→LONG) on a constant
    // After folding, the downstream op must receive a LONG constant, not FLOAT.
    // Verifies that dtype is preserved correctly through constant folding.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testConstantFolding_DtypePreservation_FloatToLong() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int seqLen = 8;
        int hiddenDim = 32;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // A FLOAT constant that will be cast to LONG by a constant-only op.
        // The cast result is a LONG constant — folding should preserve this dtype.
        INDArray floatIndices = Nd4j.linspace(0, seqLen - 1, seqLen, DataType.FLOAT).reshape(seqLen);
        SDVariable floatConst = sd.constant("cf_float_indices", floatIndices);

        // Cast to LONG — all inputs are constant, so this should be foldable
        SDVariable longIndices = floatConst.castTo("cf_long_indices", DataType.INT64);

        // Use the LONG indices in a gather operation on a constant embedding table
        INDArray embedTable = Nd4j.randn(DataType.FLOAT, seqLen, hiddenDim).mul(0.02);
        SDVariable embedConst = sd.constant("cf_embed_table", embedTable);
        SDVariable gathered = sd.gather("cf_gathered", embedConst, longIndices, 0);

        // Expand and add to input
        SDVariable expanded = sd.expandDims("cf_embed_expand", gathered, 0);
        SDVariable output = input.add("output", expanded);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Apply constant folding — cast(float_const) → long const should be folded
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ConstantFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Dtype-preserving fold output is null");
        assertFalse(optOutput.isNaN().any(), "Dtype-preserving fold output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Constant folding dtype preservation maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Constant folding changed dtype behavior: maxDiff=" + maxDiff);

        // If the optimizer folded cf_long_indices, verify it kept LONG dtype
        INDArray foldedLong = optimized.getConstantArrays().getArray("cf_long_indices");
        if (foldedLong != null) {
            assertEquals(DataType.INT64, foldedLong.dataType(),
                    "Folded LONG constant has wrong dtype: " + foldedLong.dataType());
            for (int i = 0; i < seqLen; i++) {
                assertEquals(i, foldedLong.getLong(i),
                        "Folded cf_long_indices[" + i + "] wrong: expected " + i +
                        " got " + foldedLong.getLong(i));
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // FP16 quantization with mixed constant sizes
    // Large 2D weights (>= 1024 elements) must be quantized to HALF.
    // Small 1D biases (< 1024 elements) must remain FLOAT.
    // Tests that the size threshold is applied correctly.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testFP16_MixedConstantSizes_LargeQuantizedSmallRetained() {
        Nd4j.getRandom().setSeed(42);
        SameDiff sd = SameDiff.create();

        int seqLen = 16;
        // 64*64=4096 elements: large weight gets quantized; bias is 1D length 64 < 1024, stays FLOAT
        int hiddenDim = 64;

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, seqLen, hiddenDim);

        // Large 2D weight: [64, 64] = 4096 elements → should be quantized to HALF
        INDArray largeWArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable largeW = sd.constant("mcs_large_weight", largeWArr);

        // Small 1D bias: [64] = 64 elements → must NOT be quantized (stays FLOAT)
        INDArray smallBiasArr = Nd4j.randn(DataType.FLOAT, hiddenDim).mul(0.01);
        SDVariable smallBias = sd.constant("mcs_small_bias", smallBiasArr);

        // Scalar (1 element) → must NOT be quantized
        INDArray scalarArr = Nd4j.scalar(DataType.FLOAT, 0.5f);
        SDVariable scalarConst = sd.constant("mcs_scalar", scalarArr);

        // Compute: (input @ large_weight + small_bias) * scalar
        SDVariable mm = sd.mmul("mcs_matmul", input, largeW);
        SDVariable withBias = mm.add("mcs_with_bias", smallBias);
        SDVariable output = withBias.mul("output", scalarConst);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Apply FP16 quantization
        System.setProperty("nd4j.optimizer.fp16", "true");
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new QuantizationOptimizations()));

        // Verify large_weight was quantized to HALF
        INDArray optLargeW = optimized.getConstantArrays().getArray("mcs_large_weight");
        assertNotNull(optLargeW, "mcs_large_weight missing from optimized graph");
        assertEquals(DataType.HALF, optLargeW.dataType(),
                "mcs_large_weight should be HALF after FP16 quantization, got: " + optLargeW.dataType());

        // Verify small_bias stayed FLOAT (< 1024 elements, 1D)
        INDArray optSmallBias = optimized.getConstantArrays().getArray("mcs_small_bias");
        assertNotNull(optSmallBias, "mcs_small_bias missing from optimized graph");
        assertEquals(DataType.FLOAT, optSmallBias.dataType(),
                "mcs_small_bias should remain FLOAT (< 1024 elements), got: " + optSmallBias.dataType());

        // Verify scalar stayed FLOAT (1 element)
        INDArray optScalar = optimized.getConstantArrays().getArray("mcs_scalar");
        assertNotNull(optScalar, "mcs_scalar missing from optimized graph");
        assertEquals(DataType.FLOAT, optScalar.dataType(),
                "mcs_scalar should remain FLOAT (1 element), got: " + optScalar.dataType());

        // Verify output values are close despite FP16 weight
        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "FP16 mixed-size output is null");
        assertFalse(optOutput.isNaN().any(), "FP16 mixed-size output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("FP16 mixed-size constants maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1.0, "FP16 mixed-size constants diverged: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Shape fusion: self-inverse permute composes to identity
    // permute(0,2,1,3) followed by permute(0,2,1,3) is its own inverse
    // (applying it twice returns to the original layout).
    // FuseChainedPermutes should compose them and detect the identity,
    // removing both permutes. The output must equal the original input layout.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testShapeFusion_SelfInversePermute_ComposesToIdentity() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seq = 8;
        int heads = 4;
        int headDim = 16;

        // [B, S, H, D] layout
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seq, heads, headDim);

        // permute(0,2,1,3): swap S and H — [B,S,H,D] → [B,H,S,D]
        SDVariable p1 = sd.permute("si_perm_forward", input, 0, 2, 1, 3);
        // permute(0,2,1,3) again: self-inverse — [B,H,S,D] → [B,S,H,D]
        SDVariable p2 = sd.permute("si_perm_inverse", p1, 0, 2, 1, 3);

        // A downstream op that uses the (should-be-identity) permuted result
        INDArray wArr = Nd4j.randn(DataType.FLOAT, headDim, headDim).mul(0.02);
        SDVariable w = sd.constant("si_proj_w", wArr);
        // Reshape to [B, S*H, D] for mmul, then reshape back
        SDVariable flat = sd.reshape("si_flat", p2, batch, seq * heads, headDim);
        SDVariable mm = sd.mmul("si_mm", flat, w);
        SDVariable output = sd.reshape("output", mm, batch, seq, heads, headDim);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seq, heads, headDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");

        // Apply shape fusion — should detect and remove both self-inverse permutes
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new ShapeFunctionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Self-inverse permute output is null");
        assertFalse(optOutput.isNaN().any(), "Self-inverse permute output has NaN");
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Self-inverse permute identity fusion maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Self-inverse permute fusion corrupted output: maxDiff=" + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Attention fusion with additive causal mask
    // AttentionFusionOptimizations must handle additive masks (large negative
    // values in upper triangle, zeros in lower triangle) in addition to
    // multiplicative masks. The causal mask forces future tokens to -inf before
    // softmax, so they get near-zero attention weight.
    // After fusion, the attention must still respect the causal structure.
    // ──────────────────────────────────────────────────────────────────────────

    @Test
    public void testAttentionFusion_WithCausalMask() {
        Nd4j.getRandom().setSeed(42);
        System.setProperty("nd4j.optimizer.fp16", "false");
        SameDiff sd = SameDiff.create();

        int batch = 1;
        int seqLen = 8;
        int hiddenDim = 32;
        int headDim = 8;
        int numHeads = hiddenDim / headDim; // 4

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenDim);

        // QKV projections
        INDArray wqArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wkArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        INDArray wvArr = Nd4j.randn(DataType.FLOAT, hiddenDim, hiddenDim).mul(0.02);
        SDVariable wq = sd.constant("causal_wq", wqArr);
        SDVariable wk = sd.constant("causal_wk", wkArr);
        SDVariable wv = sd.constant("causal_wv", wvArr);

        SDVariable q = sd.mmul("causal_q", input, wq);
        SDVariable k = sd.mmul("causal_k", input, wk);
        SDVariable v = sd.mmul("causal_v", input, wv);

        // Reshape to multi-head: [B, S, H, D] then permute
        SDVariable qr = sd.reshape("causal_qr", q, batch, seqLen, numHeads, headDim);
        SDVariable qt = sd.permute("causal_qt", qr, 0, 2, 1, 3);   // [B, H, S, D]
        SDVariable kr = sd.reshape("causal_kr", k, batch, seqLen, numHeads, headDim);
        SDVariable kt = sd.permute("causal_kt", kr, 0, 2, 3, 1);   // [B, H, D, S] for K^T
        SDVariable vr = sd.reshape("causal_vr", v, batch, seqLen, numHeads, headDim);
        SDVariable vt = sd.permute("causal_vt", vr, 0, 2, 1, 3);   // [B, H, S, D]

        // Q @ K^T scaled: [B, H, S, S]
        SDVariable scores = sd.mmul("causal_scores", qt, kt);
        SDVariable scale = sd.constant("causal_scale",
                Nd4j.scalar(DataType.FLOAT, 1.0f / (float) Math.sqrt(headDim)));
        SDVariable scaledScores = scores.mul("causal_scaled", scale);

        // Additive causal mask: lower-triangular zeros, upper-triangular -1e9
        // Shape [1, 1, seqLen, seqLen] — broadcasts over batch and head dims
        INDArray causalMaskArr = Nd4j.zeros(DataType.FLOAT, 1, 1, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                causalMaskArr.putScalar(new int[]{0, 0, i, j}, -1e9f);
            }
        }
        SDVariable causalMask = sd.constant("causal_mask", causalMaskArr);

        // Add causal mask to scaled scores (additive masking — standard GPT/LLaMA pattern)
        SDVariable maskedScores = scaledScores.add("causal_masked", causalMask);

        // Softmax and weighted sum
        SDVariable attnWeights = sd.nn.softmax("causal_softmax", maskedScores, -1);
        SDVariable attnOut = sd.mmul("causal_attn_out", attnWeights, vt);

        // Reshape back: [B, H, S, D] → [B, S, H, D] → [B, S, hiddenDim]
        SDVariable attnPerm = sd.permute("causal_out_perm", attnOut, 0, 2, 1, 3);
        SDVariable output = sd.reshape("output", attnPerm, batch, seqLen, hiddenDim);

        sd.setOutputs(Collections.singletonList("output"));

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenDim);
        INDArray refOutput = sd.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(refOutput);
        assertFalse(refOutput.isNaN().any(), "Reference has NaN");
        double refMax = refOutput.amaxNumber().doubleValue();
        assertTrue(refMax > 1e-6, "Reference is all zeros");

        // Apply attention fusion — must handle additive causal mask pattern
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.singletonList("output"),
                Collections.singletonList(new AttentionFusionOptimizations()));

        INDArray optOutput = optimized.outputSingle(Map.of("input", inputArr), "output");

        assertNotNull(optOutput, "Causal attention fusion output is null");
        assertFalse(optOutput.isNaN().any(), "Causal attention fusion output has NaN");
        double optMax = optOutput.amaxNumber().doubleValue();
        assertTrue(optMax > 1e-6, "Causal attention fusion output is all zeros, max=" + optMax);
        assertArrayEquals(refOutput.shape(), optOutput.shape(), "Shape mismatch");

        double maxDiff = refOutput.sub(optOutput).amaxNumber().doubleValue();
        log.info("Causal attention fusion maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-3, "Causal attention fusion corrupted output: maxDiff=" + maxDiff);
    }

    /**
     * Direct test of DotProductAttentionV2 with isCausal=true against manual computation.
     * No optimizer, no graph copy — just verify the fused op produces correct results.
     */
    @Test
    public void testDotProductAttentionV2_CausalDirect() {
        Nd4j.getRandom().setSeed(42);
        int B = 1, S = 4, H = 2, D = 4;
        float scale = (float)(1.0 / Math.sqrt(D));

        // BSHD inputs
        INDArray Q = Nd4j.randn(DataType.FLOAT, B, S, H, D);
        INDArray K = Nd4j.randn(DataType.FLOAT, B, S, H, D);
        INDArray V = Nd4j.randn(DataType.FLOAT, B, S, H, D);

        // Manual attention: permute to BHSD, compute Q@K^T*scale, causal mask, softmax, @V, permute back
        INDArray Qp = Q.permute(0, 2, 1, 3);  // [B,H,S,D]
        INDArray Kp = K.permute(0, 2, 1, 3);  // [B,H,S,D]
        INDArray Vp = V.permute(0, 2, 1, 3);  // [B,H,S,D]

        // Q@K^T per head: reshape to [B*H,S,D]
        INDArray Qf = Qp.reshape(B*H, S, D);
        INDArray Kf = Kp.reshape(B*H, S, D);
        INDArray Vf = Vp.reshape(B*H, S, D);

        // Batched matmul: Q@K^T
        INDArray scores = Nd4j.create(DataType.FLOAT, B*H, S, S);
        for (int i = 0; i < B*H; i++) {
            INDArray qi = Qf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            INDArray ki = Kf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            scores.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i)).assign(qi.mmul(ki.transpose()).mul(scale));
        }

        // Causal mask
        for (int b = 0; b < B*H; b++) {
            for (int i = 0; i < S; i++) {
                for (int j = i + 1; j < S; j++) {
                    scores.putScalar(new int[]{b, i, j}, -1e9f);
                }
            }
        }

        // Softmax on last dim
        INDArray weights = Nd4j.create(DataType.FLOAT, B*H, S, S);
        for (int b = 0; b < B*H; b++) {
            for (int i = 0; i < S; i++) {
                INDArray row = scores.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                double max = row.maxNumber().doubleValue();
                INDArray expRow = Nd4j.math.exp(row.sub(max));
                double sum = expRow.sumNumber().doubleValue();
                weights.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()).assign(expRow.div(sum));
            }
        }

        // Weighted sum: weights @ V
        INDArray out = Nd4j.create(DataType.FLOAT, B*H, S, D);
        for (int i = 0; i < B*H; i++) {
            INDArray wi = weights.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            INDArray vi = Vf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            out.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i)).assign(wi.mmul(vi));
        }

        // Permute back: [B*H,S,D] -> [B,H,S,D] -> [B,S,H,D]
        INDArray manualOut = out.reshape(B, H, S, D).permute(0, 2, 1, 3);  // BSHD

        // Run fused op with isCausal=true
        INDArray fusedResultCausal = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(
                Q, V, K, null, null, scale, 0.0, true, false))[0];

        System.out.println("[DIRECT-TEST] manual shape=" + java.util.Arrays.toString(manualOut.shape()));
        System.out.println("[DIRECT-TEST] fused causal shape=" + java.util.Arrays.toString(fusedResultCausal.shape()));
        double causalDiff = manualOut.sub(fusedResultCausal).amaxNumber().doubleValue();
        System.out.println("[DIRECT-TEST] causal maxDiff=" + causalDiff);

        // Also test without causal to confirm non-causal path works
        // Manual without causal mask
        INDArray scoresNoCausal = Nd4j.create(DataType.FLOAT, B*H, S, S);
        for (int i = 0; i < B*H; i++) {
            INDArray qi = Qf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            INDArray ki = Kf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            scoresNoCausal.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i)).assign(qi.mmul(ki.transpose()).mul(scale));
        }
        INDArray weightsNC = Nd4j.create(DataType.FLOAT, B*H, S, S);
        for (int b = 0; b < B*H; b++) {
            for (int i = 0; i < S; i++) {
                INDArray row = scoresNoCausal.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                double max = row.maxNumber().doubleValue();
                INDArray expRow = Nd4j.math.exp(row.sub(max));
                double sum = expRow.sumNumber().doubleValue();
                weightsNC.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                        org.nd4j.linalg.indexing.NDArrayIndex.point(i),
                        org.nd4j.linalg.indexing.NDArrayIndex.all()).assign(expRow.div(sum));
            }
        }
        INDArray outNC = Nd4j.create(DataType.FLOAT, B*H, S, D);
        for (int i = 0; i < B*H; i++) {
            INDArray wi = weightsNC.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            INDArray vi = Vf.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i));
            outNC.get(org.nd4j.linalg.indexing.NDArrayIndex.point(i)).assign(wi.mmul(vi));
        }
        INDArray manualNoCausal = outNC.reshape(B, H, S, D).permute(0, 2, 1, 3);

        INDArray fusedResultNoCausal = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(
                Q, V, K, null, null, scale, 0.0, false, false))[0];
        double noCausalDiff = manualNoCausal.sub(fusedResultNoCausal).amaxNumber().doubleValue();
        System.out.println("[DIRECT-TEST] non-causal maxDiff=" + noCausalDiff);

        assertTrue(noCausalDiff < 1e-4, "Non-causal DotProductAttentionV2 mismatch: maxDiff=" + noCausalDiff);
        assertTrue(causalDiff < 1e-4, "Causal DotProductAttentionV2 mismatch: maxDiff=" + causalDiff);
    }
}
