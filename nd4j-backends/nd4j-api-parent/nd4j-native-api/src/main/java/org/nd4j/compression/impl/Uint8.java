package org.nd4j.compression.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Compressor implementation based on uint8 as storage for integral values.
 * So, all data will be stored in 0..255 space, so this compressor might be useful for images only
 *
 * @author raver119@gmail.com
 */
public class Uint8 extends AbstractCompressor {
    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "UINT8";
    }

    /**
     * This method returns compression type provided by specific NDArrayCompressor implementation
     *
     * @return
     */
    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSY;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.UINT8, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer, DataBuffer.TypeEx.UINT8);
        return result;
    }
}
