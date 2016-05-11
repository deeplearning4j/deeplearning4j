package org.nd4j.jita.allocator.tad;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class DeviceTADManager extends BasicTADManager {
    @Override
    public DataBuffer getTADOnlyShapeInfo(INDArray array, int[] dimension, int dimensionLength) {
        return super.getTADOnlyShapeInfo(array, dimension, dimensionLength);
    }
}
