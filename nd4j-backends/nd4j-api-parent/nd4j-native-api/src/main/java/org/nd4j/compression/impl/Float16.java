package org.nd4j.compression.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Compressor implementation based on half-precision floats, aka FP16
 *
 * @author raver119@
 */
public class Float16 extends AbstractCompressor  {

    @Override
    public String getDescriptor() {
        return "FLOAT16";
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
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.FLOAT16, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(getLocalTypeEx(buffer), buffer, DataBuffer.TypeEx.FLOAT16);
        return result;
    }
}
