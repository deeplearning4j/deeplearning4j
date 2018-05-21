package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Audrey Loeffel
 */
abstract public class BaseSparseInfoProvider implements SparseInfoProvider {
    @Override
    public DataBuffer createSparseInformation(int[] flags, long[] sparseOffsets, int[] hiddenDimensions,
                    int underlyingRank) {
        return createSparseInformation(flags, sparseOffsets, hiddenDimensions, underlyingRank);
    }
}
