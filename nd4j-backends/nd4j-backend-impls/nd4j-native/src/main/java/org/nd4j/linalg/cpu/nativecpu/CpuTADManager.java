package org.nd4j.linalg.cpu.nativecpu;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TADManager;

/**
 * @author raver119@gmail.com
 */
public class CpuTADManager implements TADManager {
    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        return null;
    }
}
