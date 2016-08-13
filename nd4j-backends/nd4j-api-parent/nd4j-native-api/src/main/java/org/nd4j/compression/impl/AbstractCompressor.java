package org.nd4j.compression.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.NDArrayCompressor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author raver119@gmail.com
 */
public abstract class AbstractCompressor implements NDArrayCompressor {
    protected static Logger logger = LoggerFactory.getLogger(AbstractCompressor.class);

    @Override
    public INDArray compress(INDArray array) {
        INDArray dup = array.dup();
        dup.setData(compress(dup.data()));

        return dup;
    }



    @Override
    public INDArray decompress(INDArray array) {
        DataBuffer buffer = decompress(array.data());
        DataBuffer shapeInfo = array.shapeInfoDataBuffer();
        INDArray rest = Nd4j.createArrayFromShapeBuffer(buffer, shapeInfo);

        return rest;
    }

    public abstract DataBuffer decompress(DataBuffer buffer);
    public abstract DataBuffer compress(DataBuffer buffer);
}
