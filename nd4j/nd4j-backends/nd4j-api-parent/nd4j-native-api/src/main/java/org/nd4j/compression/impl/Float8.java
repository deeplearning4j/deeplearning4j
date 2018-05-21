package org.nd4j.compression.impl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Compressor implementation based on 8-bit floats, aka FP8 or Quarter
 *
 * @author raver119@gmail.com
 */
public class Float8 extends AbstractCompressor {
    @Override
    public String getDescriptor() {
        return "FLOAT8";
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
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.FLOAT8, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer,
                        DataBuffer.TypeEx.FLOAT8);
        return result;
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataBuffer.TypeEx srcType, Pointer srcPointer, int length,
                    int elementSize) {

        BytePointer ptr = new BytePointer(length);
        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(length * 1);
        descriptor.setOriginalLength(length * elementSize);
        descriptor.setOriginalElementSize(elementSize);
        descriptor.setNumberOfElements(length);

        descriptor.setCompressionAlgorithm(getDescriptor());
        descriptor.setCompressionType(getCompressionType());

        CompressedDataBuffer buffer = new CompressedDataBuffer(ptr, descriptor);

        Nd4j.getNDArrayFactory().convertDataEx(srcType, srcPointer, DataBuffer.TypeEx.FLOAT8, ptr, length);

        return buffer;
    }
}
