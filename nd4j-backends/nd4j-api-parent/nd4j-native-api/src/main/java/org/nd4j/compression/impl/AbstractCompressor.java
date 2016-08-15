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
        INDArray dup = array.dup(array.ordering());
        dup.setData(compress(dup.data()));
        dup.markAsCompressed(true);

        return dup;
    }

    /**
     * Inplace compression of INDArray
     *
     * @param array
     */
    @Override
    public void compressi(INDArray array) {
        // TODO: lift this restriction
        if (array.isView())
            throw new UnsupportedOperationException("Impossible to apply inplace compression on View");

        array.setData(compress(array.data()));
        array.markAsCompressed(true);
    }

    @Override
    public void decompressi(INDArray array) {
        if (!array.isCompressed())
            return;

        array.markAsCompressed(false);
        array.setData(decompress(array.data()));
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

    protected DataBuffer.TypeEx convertType(DataBuffer.Type type) {
        if (type == DataBuffer.Type.FLOAT) {
            return DataBuffer.TypeEx.FLOAT;
        } else if (type == DataBuffer.Type.DOUBLE) {
            return DataBuffer.TypeEx.DOUBLE;
        } else
            throw new IllegalStateException("Unknown dataType: [" + type + "]");
    }

    protected DataBuffer.TypeEx getGlobalTypeEx() {
        DataBuffer.Type type = Nd4j.dataType();

        return convertType(type);
    }

    protected DataBuffer.TypeEx getLocalTypeEx(DataBuffer buffer) {
        DataBuffer.Type type = buffer.dataType();

        return convertType(type);
    }
}
