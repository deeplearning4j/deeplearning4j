package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that reshape with view creation handles non-contiguous source arrays correctly.
 * This reproduces the DSP attention divergence: DSP provides arrays with non-standard
 * strides from view ops (permute, etc.), and reshape('c', ..., false) creates views
 * that may read data in the wrong order.
 */
@Slf4j
public class TestReshapeViewStrides {

    @Test
    @DisplayName("reshape [1,17,576] -> [1,17,9,64] on C-contiguous array")
    public void testReshapeCContiguous() {
        // Standard case: C-contiguous [1,17,576]
        INDArray arr = Nd4j.linspace(1, 17 * 576, 17 * 576, DataType.FLOAT).reshape(1, 17, 576);
        log.info("C-contiguous: shape={} strides={} ordering={}",
                Arrays.toString(arr.shape()), Arrays.toString(arr.stride()), arr.ordering());

        INDArray reshaped = arr.reshape('c', 1, 17, 9, 64);
        log.info("Reshaped: shape={} strides={}", Arrays.toString(reshaped.shape()), Arrays.toString(reshaped.stride()));

        // Element [0,9,2,47] should be at flat offset 9*576 + 2*64 + 47 = 5184 + 128 + 47 = 5359
        float expected = arr.getFloat(0, 9, 2 * 64 + 47);  // hidden dim 175
        float actual = reshaped.getFloat(0, 9, 2, 47);
        log.info("C-contiguous: arr[0,9,175]={} reshaped[0,9,2,47]={}", expected, actual);
        assertEquals(expected, actual, 1e-6, "C-contiguous reshape should preserve element access");
    }

    @Test
    @DisplayName("reshape [1,17,576] -> [1,17,9,64] on permuted (non-contiguous) array")
    public void testReshapePermuted() {
        // Create a permuted array: [17,1,576] permuted to [1,17,576]
        // This has strides [576, 9792, 1] — NOT C-contiguous for [1,17,576]
        INDArray base = Nd4j.linspace(1, 17 * 576, 17 * 576, DataType.FLOAT).reshape(17, 1, 576);
        INDArray permuted = base.permute(1, 0, 2);  // [1,17,576] with strides [576, 9792, 1]
        // No — permute of [17,1,576] with {1,0,2} gives [1,17,576] with strides from base
        // base strides for [17,1,576] in C: [576, 576, 1]. permute(1,0,2) → strides [576, 576, 1]
        // That's still C-contiguous. Let me create a truly non-contiguous one.

        // Better: create [576,17] F-order, transpose to [17,576], add batch dim
        INDArray fOrder = Nd4j.linspace(1, 17 * 576, 17 * 576, DataType.FLOAT)
                .reshape('f', 576, 17);  // F-order [576,17]
        INDArray transposed = fOrder.transpose();  // [17,576] — non-contiguous view of F-order data
        INDArray withBatch = transposed.reshape(1, 17, 576);  // may or may not be contiguous

        log.info("F-transposed: shape={} strides={} ordering={}",
                Arrays.toString(withBatch.shape()), Arrays.toString(withBatch.stride()), withBatch.ordering());

        // Now reshape to [1,17,9,64]
        INDArray reshaped = withBatch.reshape('c', 1, 17, 9, 64);
        log.info("Reshaped from non-contig: shape={} strides={}",
                Arrays.toString(reshaped.shape()), Arrays.toString(reshaped.stride()));

        // Compare element access
        float fromFlat = withBatch.getFloat(0, 9, 175);  // hidden dim 175
        float fromReshaped = reshaped.getFloat(0, 9, 2, 47);  // head 2, dim 47

        log.info("Non-contiguous: withBatch[0,9,175]={} reshaped[0,9,2,47]={}", fromFlat, fromReshaped);
        assertEquals(fromFlat, fromReshaped, 1e-6,
                "Reshape of non-contiguous array must preserve element access. " +
                "withBatch strides=" + Arrays.toString(withBatch.stride()));
    }

    @Test
    @DisplayName("DSP-like scenario: matmul output with view strides then reshape")
    public void testDspViewReshape() {
        // Simulate what DSP does: allocate [1,17,576], write via an op,
        // then the result may be a view of a larger buffer or have non-standard strides.
        // The simplest non-contiguous case: create from a slice of a larger array.
        INDArray large = Nd4j.rand(DataType.FLOAT, 2, 17, 576);
        INDArray slice = large.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        // slice is [17,576] — a view into large with offset, may have non-standard strides
        INDArray withBatch = slice.reshape(1, 17, 576);

        log.info("Slice view: shape={} strides={} ordering={} isView={}",
                Arrays.toString(withBatch.shape()), Arrays.toString(withBatch.stride()),
                withBatch.ordering(), withBatch.isView());

        // Reshape to [1,17,9,64]
        INDArray reshaped = withBatch.reshape('c', 1, 17, 9, 64);

        // Compare: element [0,9,175] in 3D should equal [0,9,2,47] in 4D
        float from3d = withBatch.getFloat(0, 9, 175);
        float from4d = reshaped.getFloat(0, 9, 2, 47);

        log.info("DSP-like: withBatch[0,9,175]={} reshaped[0,9,2,47]={}", from3d, from4d);
        assertEquals(from3d, from4d, 1e-6,
                "Reshape of view array must preserve element access. " +
                "withBatch strides=" + Arrays.toString(withBatch.stride()) +
                " reshaped strides=" + Arrays.toString(reshaped.stride()));
    }

    @Test
    @DisplayName("Contiguous dup vs original: reshape should give same values")
    public void testDupVsOriginalReshape() {
        // Create non-contiguous array, reshape both the original and a dup'd copy
        INDArray large = Nd4j.rand(DataType.FLOAT, 2, 17, 576);
        INDArray slice = large.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all())
                .reshape(1, 17, 576);

        INDArray contiguous = slice.dup('c');

        INDArray reshapedOrig = slice.reshape('c', 1, 17, 9, 64);
        INDArray reshapedDup = contiguous.reshape('c', 1, 17, 9, 64);

        log.info("Original strides: {}", Arrays.toString(slice.stride()));
        log.info("Dup strides: {}", Arrays.toString(contiguous.stride()));
        log.info("Reshaped-orig strides: {}", Arrays.toString(reshapedOrig.stride()));
        log.info("Reshaped-dup strides: {}", Arrays.toString(reshapedDup.stride()));

        // ALL elements should match
        double maxDiff = Nd4j.math().abs(reshapedOrig.sub(reshapedDup)).maxNumber().doubleValue();
        log.info("maxDiff between reshape(original) and reshape(dup): {}", maxDiff);

        // Specifically check the divergent position
        float origVal = reshapedOrig.getFloat(0, 9, 2, 47);
        float dupVal = reshapedDup.getFloat(0, 9, 2, 47);
        log.info("At [0,9,2,47]: original={} dup={}", origVal, dupVal);

        assertEquals(0.0, maxDiff, 1e-5,
                "Reshape of view vs reshape of dup must produce identical values. maxDiff=" + maxDiff);
    }
}
