package org.nd4j.compression.impl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Compressor implementation based on uint8 as storage for integral values.
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
        if (buffer.dataType() != DataBuffer.Type.COMPRESSED)
            throw new RuntimeException("Unsupported source dataType: " + buffer.dataType());

        CompressedDataBuffer comp = (CompressedDataBuffer) buffer;
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();

        DataBuffer result = Nd4j.createBuffer(descriptor.getCompressedLength());
        UByteIndexer indexer = UByteIndexer.create((BytePointer) comp.getPointer());

        for (int i = 0; i < result.length(); i++) {
            result.put(i, indexer.get(i));
        }

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        BytePointer pointer = new BytePointer(buffer.length());
        UByteIndexer indexer = UByteIndexer.create(pointer);

        for (int x = 0; x < buffer.length(); x ++) {
            int t = (int) buffer.getDouble(x);

            if (t > 254) t = 255;
            if (t < 0) t = 0;

            indexer.put(x, t);
        }

        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressionAlgorithm(getDescriptor());
        descriptor.setOriginalLength(buffer.length() * buffer.getElementSize());
        descriptor.setCompressedLength(buffer.length());
        descriptor.setCompressionType(CompressionType.LOSSY);

        CompressedDataBuffer result = new CompressedDataBuffer(pointer, descriptor);

        return result;
    }
}
