package org.nd4j.compression.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.NDArrayCompressor;

/**
 * @author raver119@gmail.com
 */
public abstract class AbstractCompressor implements NDArrayCompressor {

    @Override
    public INDArray compress(INDArray array) {
        return null;
    }



    @Override
    public INDArray decompress(INDArray array) {
        return null;
    }

    public abstract DataBuffer decompress(DataBuffer buffer);
    public abstract DataBuffer compress(DataBuffer buffer);
}
