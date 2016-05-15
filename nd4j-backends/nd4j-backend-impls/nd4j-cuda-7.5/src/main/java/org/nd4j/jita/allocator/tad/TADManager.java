package org.nd4j.jita.allocator.tad;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface TADManager {
    Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension);
}
