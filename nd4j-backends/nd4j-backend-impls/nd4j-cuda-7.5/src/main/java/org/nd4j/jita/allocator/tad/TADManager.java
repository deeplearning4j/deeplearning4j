package org.nd4j.jita.allocator.tad;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface TADManager {
    DataBuffer getTADOnlyShapeInfo(INDArray array, int[] dimension);
}
