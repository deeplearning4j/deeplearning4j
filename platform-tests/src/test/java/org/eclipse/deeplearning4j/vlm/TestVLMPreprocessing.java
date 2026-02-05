package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Test that reduce ops (min, max, sum) work correctly on multi-dimensional arrays.
 * Regression test for bug where -1 sentinel for "reduce all" was being interpreted
 * as "last axis" by normalizeAxis(), causing reduce ops on rank>1 arrays to only
 * reduce along the last dimension.
 */
@Slf4j
public class TestVLMPreprocessing {

    @Test
    public void testReduceAllDimensions() {
        float[] data = {-1, -1, -1, 1, 1, 1, 2, 3};  // min=-1, max=3, sum=5

        // 1D - baseline
        INDArray s1d = Nd4j.create(data);
        assertEquals(-1.0, s1d.minNumber().doubleValue(), 1e-5, "1D min");
        assertEquals(3.0, s1d.maxNumber().doubleValue(), 1e-5, "1D max");
        assertEquals(5.0, s1d.sumNumber().doubleValue(), 1e-5, "1D sum");

        // 2D [2,4] - same data, different shape
        INDArray s2d = Nd4j.create(data, new int[]{2, 4});
        assertEquals(-1.0, s2d.minNumber().doubleValue(), 1e-5, "2D [2,4] min");
        assertEquals(3.0, s2d.maxNumber().doubleValue(), 1e-5, "2D [2,4] max");
        assertEquals(5.0, s2d.sumNumber().doubleValue(), 1e-5, "2D [2,4] sum");

        // 2D [4,2]
        INDArray s2d2 = Nd4j.create(data, new int[]{4, 2});
        assertEquals(-1.0, s2d2.minNumber().doubleValue(), 1e-5, "2D [4,2] min");
        assertEquals(3.0, s2d2.maxNumber().doubleValue(), 1e-5, "2D [4,2] max");
        assertEquals(5.0, s2d2.sumNumber().doubleValue(), 1e-5, "2D [4,2] sum");

        // 3D [2,2,2]
        INDArray s3d = Nd4j.create(data, new int[]{2, 2, 2});
        assertEquals(-1.0, s3d.minNumber().doubleValue(), 1e-5, "3D [2,2,2] min");
        assertEquals(3.0, s3d.maxNumber().doubleValue(), 1e-5, "3D [2,2,2] max");
        assertEquals(5.0, s3d.sumNumber().doubleValue(), 1e-5, "3D [2,2,2] sum");

        // 4D [1,2,2,2]
        INDArray s4d = Nd4j.create(data, new int[]{1, 2, 2, 2});
        assertEquals(-1.0, s4d.minNumber().doubleValue(), 1e-5, "4D [1,2,2,2] min");
        assertEquals(3.0, s4d.maxNumber().doubleValue(), 1e-5, "4D [1,2,2,2] max");
        assertEquals(5.0, s4d.sumNumber().doubleValue(), 1e-5, "4D [1,2,2,2] sum");

        // Large 4D array [1,3,512,512] with known values
        int n = 786432;
        float[] large = new float[n];
        for (int i = 0; i < n / 2; i++) large[i] = -1.0f;
        for (int i = n / 2; i < n; i++) large[i] = 1.0f;

        INDArray l4d = Nd4j.create(large, new int[]{1, 3, 512, 512});
        assertEquals(-1.0, l4d.minNumber().doubleValue(), 1e-5, "Large 4D min");
        assertEquals(1.0, l4d.maxNumber().doubleValue(), 1e-5, "Large 4D max");
        assertEquals(0.0, l4d.sumNumber().doubleValue(), 1e-5, "Large 4D sum");

        log.info("All reduce assertions passed");
    }
}
