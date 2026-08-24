/*
 * Non-contiguous view rms_norm / layer_norm CPU correctness test.
 *
 * Reproduces the SmolDocling vision-encoder divergence:
 * rms_norm fed a permuted (non-contiguous) view gives wrong output on CPU because
 * rmsNorm_() used `x + row * rowLen` offsets, which assumes contiguous layout.
 * For a permuted view the actual row stride is strideAt(-2), NOT rowLen.
 */
package org.eclipse.deeplearning4j.nd4j.linalg.cpu;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class NonContigViewLayerNormTest {

    /**
     * Creates a permuted (non-contiguous) view and runs rms_norm.
     * Reference is computed on a materialised contiguous copy with the same values.
     *
     * Shapes chosen to match the SmolDocling vision encoder:
     *   base:     [1, 768, 576]  (original output of some conv/linear)
     *   permuted: [1, 576, 768]  (after permute(0, 2, 1))
     * After permute, strideAt(1)=1, strideAt(2)=576 — the rmsNorm_ row offset
     * "x + row * 768" is wrong; the correct offset is "x + row * 1".
     *
     * Use smaller numbers for speed: [1, 8, 4] -> permuted [1, 4, 8].
     * rowLen=8, but after permute stride(1)=1 instead of 8.
     */
    @Test
    public void testRmsNormOnPermutedView() {
        // base [1, C, H] = [1, 4, 8]: rows in memory are of length H=8 with stride C*H=32 between
        long B = 1L, C = 4L, H = 8L;
        float[] data = new float[(int)(B * C * H)];
        for (int i = 0; i < data.length; i++) data[i] = (i + 1) * 0.5f;
        INDArray base = Nd4j.create(data, new long[]{B, C, H}, DataType.FLOAT);

        // Permuted view [B, H, C] = [1, 8, 4] — non-contiguous: stride(1)=1, stride(2)=H=8
        INDArray permuted = base.permute(0, 2, 1);  // shape=[1,8,4], strides=[32,1,4]
        INDArray contiguous = permuted.dup('c');      // same values, contiguous strides=[32,4,1]

        System.out.printf("permuted strides: %d, %d, %d%n",
                permuted.stride(0), permuted.stride(1), permuted.stride(2));
        System.out.printf("contiguous strides: %d, %d, %d%n",
                contiguous.stride(0), contiguous.stride(1), contiguous.stride(2));

        // Gamma for rms_norm — 1D, size = last dim = C = 4
        INDArray gamma = Nd4j.ones(DataType.FLOAT, C);

        // rms_norm on permuted (non-contiguous) view
        INDArray outPermuted = Nd4j.create(DataType.FLOAT, permuted.shape());
        DynamicCustomOp opPermuted = DynamicCustomOp.builder("rms_norm")
                .addInputs(permuted, gamma)
                .addOutputs(outPermuted)
                .build();
        Nd4j.getExecutioner().exec(opPermuted);

        // rms_norm on contiguous copy (reference)
        INDArray outContiguous = Nd4j.create(DataType.FLOAT, contiguous.shape());
        DynamicCustomOp opContiguous = DynamicCustomOp.builder("rms_norm")
                .addInputs(contiguous, gamma)
                .addOutputs(outContiguous)
                .build();
        Nd4j.getExecutioner().exec(opContiguous);

        double maxAbsDiff = outPermuted.sub(outContiguous).amaxNumber().doubleValue();
        double maxAbs = outContiguous.amaxNumber().doubleValue();
        double maxRelErr = maxAbs > 1e-8 ? maxAbsDiff / maxAbs : maxAbsDiff;

        System.out.printf("rms_norm permuted[0,0,*]: %.6f, %.6f, %.6f, %.6f%n",
                outPermuted.getFloat(0, 0, 0), outPermuted.getFloat(0, 0, 1),
                outPermuted.getFloat(0, 0, 2), outPermuted.getFloat(0, 0, 3));
        System.out.printf("rms_norm contig  [0,0,*]: %.6f, %.6f, %.6f, %.6f%n",
                outContiguous.getFloat(0, 0, 0), outContiguous.getFloat(0, 0, 1),
                outContiguous.getFloat(0, 0, 2), outContiguous.getFloat(0, 0, 3));
        System.out.printf("maxAbsDiff=%.6f  maxRelErr=%.6f%n", maxAbsDiff, maxRelErr);

        assertTrue(maxRelErr < 1e-4,
                "rms_norm on non-contiguous (permuted) view gave wrong result: maxRelErr=" + maxRelErr
                        + " (expected < 1e-4). This is the CPU non-contiguous view bug.");
    }

    /**
     * Same test but with larger shapes matching the actual vision encoder dimensions:
     * [1, 576, 768] permuted from [1, 768, 576].
     * This uses a larger array to make any stride bug unmissable.
     */
    @Test
    public void testRmsNormOnPermutedViewLargeShape() {
        long B = 1L, C = 64L, H = 16L;  // smaller but still representative ratio
        float[] data = new float[(int)(B * C * H)];
        for (int i = 0; i < data.length; i++) data[i] = (float)Math.sin(i * 0.1) + 1.0f;
        INDArray base = Nd4j.create(data, new long[]{B, C, H}, DataType.FLOAT);

        // Permuted [1, H, C] = [1, 16, 64], strides [C*H, 1, H] = [1024, 1, 16]
        INDArray permuted = base.permute(0, 2, 1);
        INDArray contiguous = permuted.dup('c');

        System.out.printf("large: permuted shape=[%d,%d,%d] strides=[%d,%d,%d]%n",
                permuted.size(0), permuted.size(1), permuted.size(2),
                permuted.stride(0), permuted.stride(1), permuted.stride(2));

        // Gamma 1D, size = C = 64
        INDArray gamma = Nd4j.ones(DataType.FLOAT, C);

        INDArray outPermuted = Nd4j.create(DataType.FLOAT, permuted.shape());
        DynamicCustomOp opP = DynamicCustomOp.builder("rms_norm")
                .addInputs(permuted, gamma)
                .addOutputs(outPermuted)
                .build();
        Nd4j.getExecutioner().exec(opP);

        INDArray outContig = Nd4j.create(DataType.FLOAT, contiguous.shape());
        DynamicCustomOp opC = DynamicCustomOp.builder("rms_norm")
                .addInputs(contiguous, gamma)
                .addOutputs(outContig)
                .build();
        Nd4j.getExecutioner().exec(opC);

        double maxAbsDiff = outPermuted.sub(outContig).amaxNumber().doubleValue();
        double maxAbs = outContig.amaxNumber().doubleValue();
        double maxRelErr = maxAbs > 1e-8 ? maxAbsDiff / maxAbs : maxAbsDiff;

        System.out.printf("large rms_norm: maxAbsDiff=%.6f  maxRelErr=%.6f%n", maxAbsDiff, maxRelErr);

        assertTrue(maxRelErr < 1e-4,
                "rms_norm (large) on non-contiguous view gave wrong result: maxRelErr=" + maxRelErr
                        + " (expected < 1e-4).");
    }

    /**
     * layer_norm on a permuted view (oneDNN path).
     * The oneDNN descriptor used getFormat() which returns the ordering tag (e.g. 'abc'),
     * not the actual strides. For a permuted 'c'-ordered view the strides don't match
     * the 'abc' tag, so oneDNN reads the wrong bytes.
     */
    @Test
    public void testLayerNormOnPermutedView() {
        long B = 1L, C = 4L, H = 8L;
        float[] data = new float[(int)(B * C * H)];
        for (int i = 0; i < data.length; i++) data[i] = (i + 1) * 0.1f;
        INDArray base = Nd4j.create(data, new long[]{B, C, H}, DataType.FLOAT);

        INDArray permuted = base.permute(0, 2, 1);  // [1, 8, 4]
        INDArray contiguous = permuted.dup('c');

        INDArray gain = Nd4j.ones(DataType.FLOAT, C);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, C);

        INDArray outPermuted = Nd4j.create(DataType.FLOAT, permuted.shape());
        DynamicCustomOp opP = DynamicCustomOp.builder("layer_norm")
                .addInputs(permuted, gain, bias)
                .addOutputs(outPermuted)
                .addIntegerArguments(2)
                .addBooleanArguments(false)  // NHWC: gain/bias follow the last dimension
                .build();
        Nd4j.getExecutioner().exec(opP);

        INDArray outContiguous = Nd4j.create(DataType.FLOAT, contiguous.shape());
        DynamicCustomOp opC = DynamicCustomOp.builder("layer_norm")
                .addInputs(contiguous, gain, bias)
                .addOutputs(outContiguous)
                .addIntegerArguments(2)
                .addBooleanArguments(false)  // same NHWC contract as the permuted view
                .build();
        Nd4j.getExecutioner().exec(opC);

        double maxAbsDiff = outPermuted.sub(outContiguous).amaxNumber().doubleValue();
        double maxAbs = outContiguous.amaxNumber().doubleValue();
        double maxRelErr = maxAbs > 1e-8 ? maxAbsDiff / maxAbs : maxAbsDiff;

        System.out.printf("layer_norm non-contig: maxAbsDiff=%.6f  maxRelErr=%.6f%n", maxAbsDiff, maxRelErr);
        System.out.printf("  permuted  [0,0,*]: %.6f, %.6f%n",
                outPermuted.getFloat(0, 0, 0), outPermuted.getFloat(0, 0, 1));
        System.out.printf("  contiguous[0,0,*]: %.6f, %.6f%n",
                outContiguous.getFloat(0, 0, 0), outContiguous.getFloat(0, 0, 1));

        assertTrue(maxRelErr < 1e-4,
                "layer_norm on non-contiguous (permuted) view gave wrong result: maxRelErr=" + maxRelErr
                        + " (expected < 1e-4).");
    }
}
