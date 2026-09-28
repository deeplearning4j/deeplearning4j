package org.eclipse.deeplearning4j.safetensors;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.eclipse.deeplearning4j.safetensors.ModelOptQwenConfig.require;

/**
 * NVFP4 weight-only quantization with NVIDIA ModelOpt's recipe
 * ({@code modelopt/torch/quantization/qtensor/nvfp4_tensor.py}), producing exactly the
 * storage a ModelOpt {@code W4A16_NVFP4} export carries and {@code modelopt_nvfp4_linear}
 * consumes: for a dense {@code [N, K]} weight, K a multiple of 16,
 * <ul>
 *   <li>global scale {@code g = amax(|W|) / (6 * 448)} (FLOAT32 scalar);</li>
 *   <li>per 16-element block, {@code s = amax(|block|) / (6 * g)} (1 for an all-zero block),
 *       stored as FP8 E4M3FN with round-to-nearest-even ({@code [N, K/16]});</li>
 *   <li>each element {@code w / (decode(s) * g)} rounded to the nearest E2M1 value
 *       (0, 0.5, 1, 1.5, 2, 3, 4, 6; ties to the even code, saturating at 6), sign in bit 3,
 *       two per byte with the even column in the low nibble ({@code [N, K/2]}).</li>
 * </ul>
 * All arithmetic is FLOAT32 in ModelOpt's operation order, so the result is bit-identical
 * to the reference for the same input.
 */
public final class ModelOptNvfp4Quantizer {
    /** Elements per block scale. */
    public static final int BLOCK = 16;
    private static final float E2M1_MAX = 6.0f;
    private static final float E4M3_MAX = 448.0f;
    // Values below the midpoint to the next (unrepresentable) step, 448 + 32 / 2, round to 448:
    // FP32 amax / (6 * global) can land a few ULP above 448 for the largest block.
    private static final float E4M3_ROUNDING_LIMIT = 464.0f;
    // Rounding boundaries between consecutive E2M1 magnitudes; a value equal to the
    // boundary at an odd index rounds up (to the even code), otherwise down.
    private static final float[] E2M1_BOUNDS = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    // Values of the non-negative finite E4M3FN codes 0x00..0x7E (0x7F is NaN), ascending.
    private static final float[] E4M3_VALUES = new float[127];

    static {
        for (int code = 0; code < E4M3_VALUES.length; code++) {
            int exponent = code >>> 3, mantissa = code & 7;
            E4M3_VALUES[code] = exponent == 0
                    ? Math.scalb((float) mantissa, -9)
                    : Math.scalb(1.0f + mantissa / 8.0f, exponent - 7);
        }
    }

    private ModelOptNvfp4Quantizer() { }

    /** Packed weights, block scales and global scale of one quantized projection. */
    public static final class Quantized {
        public final INDArray packed;
        public final INDArray blockScales;
        public final INDArray globalScale;

        Quantized(INDArray packed, INDArray blockScales, INDArray globalScale) {
            this.packed = packed;
            this.blockScales = blockScales;
            this.globalScale = globalScale;
        }
    }

    /** Quantizes a dense {@code [N, K]} floating-point weight (K a positive multiple of 16). */
    public static Quantized quantize(INDArray weight) {
        require(weight.rank() == 2 && weight.dataType().isFPType(), "NVFP4 quantization needs a rank-2 float weight");
        final long rows = weight.size(0), columns = weight.size(1);
        require(rows > 0 && columns > 0 && columns % BLOCK == 0, "NVFP4 quantization needs K % 16 == 0");
        require(rows * columns <= Integer.MAX_VALUE, "NVFP4 quantization input too large");
        final float[] w = weight.castTo(DataType.FLOAT).dup('c').data().asFloat();

        float amax = 0.0f;
        for (float value : w) amax = Math.max(amax, Math.abs(value));
        require(amax > 0.0f && Float.isFinite(amax), "NVFP4 quantization needs a finite nonzero weight");
        final float global = amax / (E2M1_MAX * E4M3_MAX);
        final float blockDenominator = E2M1_MAX * global;

        final int blocksPerRow = (int) (columns / BLOCK);
        final byte[] scales = new byte[(int) (rows * blocksPerRow)];
        final byte[] packed = new byte[(int) (rows * columns / 2)];
        final byte[] codes = new byte[BLOCK];
        for (int block = 0; block < scales.length; block++) {
            final int base = block * BLOCK;
            float blockAmax = 0.0f;
            for (int i = 0; i < BLOCK; i++) blockAmax = Math.max(blockAmax, Math.abs(w[base + i]));
            float blockScale = blockAmax / blockDenominator;
            if (blockScale == 0.0f) blockScale = 1.0f;
            final int scaleCode = encodeE4m3(blockScale);
            scales[block] = (byte) scaleCode;
            final float denominator = E4M3_VALUES[scaleCode] * global;
            for (int i = 0; i < BLOCK; i++) codes[i] = encodeE2m1(w[base + i] / denominator);
            for (int i = 0; i < BLOCK; i += 2) {
                packed[base / 2 + i / 2] = (byte) ((codes[i] & 0x0F) | ((codes[i + 1] & 0x0F) << 4));
            }
        }
        try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            return new Quantized(raw(DataType.UBYTE, packed, rows, columns / 2),
                    raw(DataType.FLOAT8, scales, rows, blocksPerRow),
                    Nd4j.scalar(DataType.FLOAT, global));
        }
    }

    /** E2M1 code of a scaled value: nearest magnitude, ties to the even code, saturating at 6. */
    static byte encodeE2m1(float value) {
        final float magnitude = Math.abs(value);
        int code = 0;
        for (int i = 0; i < E2M1_BOUNDS.length; i++) {
            if (magnitude > E2M1_BOUNDS[i]) code++;
            else if (magnitude == E2M1_BOUNDS[i] && (i & 1) == 1) code++;
        }
        return (byte) (value < 0.0f ? code | 8 : code);
    }

    /** E4M3FN code of a non-negative value below 464: nearest, ties to the even code. */
    static int encodeE4m3(float value) {
        require(value >= 0.0f && value < E4M3_ROUNDING_LIMIT, "Block scale outside the E4M3 range: " + value);
        int low = 0, high = E4M3_VALUES.length - 1;
        while (high - low > 1) {
            final int middle = (low + high) >>> 1;
            if (E4M3_VALUES[middle] <= value) low = middle;
            else high = middle;
        }
        final float below = value - E4M3_VALUES[low], above = E4M3_VALUES[high] - value;
        if (below < above) return low;
        if (above < below) return high;
        return (low & 1) == 0 ? low : high;
    }

    static float decodeE4m3(int code) {
        return E4M3_VALUES[code & 0x7F] * ((code & 0x80) != 0 ? -1.0f : 1.0f);
    }

    /** A typed array holding the given storage bytes (no numeric conversion). */
    private static INDArray raw(DataType dtype, byte[] bytes, long... shape) {
        INDArray array = Nd4j.createUninitialized(dtype, shape, 'c');
        // Borrow the array's host pointer; do not close/deallocate this alias.
        new BytePointer(array.data().pointer()).capacity(bytes.length).put(bytes);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }
}
