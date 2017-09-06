package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Audrey Loeffel
 */
public interface SparseInfoProvider {

    DataBuffer createSparseInformation(int[] flags, long[] sparseOffsets, int[] hiddenDimensions, int underlyingRank);

    void purgeCache();
}
