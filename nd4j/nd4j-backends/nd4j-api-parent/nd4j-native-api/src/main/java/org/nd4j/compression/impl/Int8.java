package org.nd4j.compression.impl;

import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Compressor implementation based on int8/aka char/aka byte as storage for integral values.
 * So, all data will be stored in -128..127 space
 *
 * @author raver119@gmail.com
 */
public class Int8 extends AbstractCompressor {
    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "INT8";
    }

    /**
     * This method returns compression opType provided by specific NDArrayCompressor implementation
     *
     * @return
     */
    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSY;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.INT8, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        DataBuffer result =
                        Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer, DataBuffer.TypeEx.INT8);
        return result;
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataBuffer.TypeEx srcType, Pointer srcPointer, int length,
                    int elementSize) {

        val ptr = new BytePointer(length);
        val descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(length * 1);
        descriptor.setOriginalLength(length * elementSize);
        descriptor.setOriginalElementSize(elementSize);
        descriptor.setNumberOfElements(length);

        descriptor.setCompressionAlgorithm(getDescriptor());
        descriptor.setCompressionType(getCompressionType());

        val buffer = new CompressedDataBuffer(ptr, descriptor);

        //Nd4j.getNDArrayFactory().convertDataEx(srcType, srcPointer, DataBuffer.TypeEx.INT8, ptr, length);
        Nd4j.getNDArrayFactory().convertDataEx(srcType, srcPointer, DataBuffer.TypeEx.INT8, buffer);

        return buffer;
    }
}
