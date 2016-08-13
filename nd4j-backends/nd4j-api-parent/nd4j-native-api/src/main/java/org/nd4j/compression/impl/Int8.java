package org.nd4j.compression.impl;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.indexer.ByteIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
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
        ByteIndexer indexer = ByteIndexer.create((BytePointer) comp.getPointer());

        for (int i = 0; i < result.length(); i++) {
            double t = (double) indexer.get(i);

            result.put(i, t);
        }

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        BytePointer pointer = new BytePointer(buffer.length());
        ByteIndexer indexer = ByteIndexer.create(pointer);

        for (int x = 0; x < buffer.length(); x ++) {
            int t = (int) buffer.getDouble(x);

            if (t > 127) t = 127;
            if (t < -128) t = -128;

            byte b = (byte) t;

            indexer.put(x, b);
        }

        CompressionDescriptor descriptor = new CompressionDescriptor(buffer, this);
        descriptor.setCompressedLength(buffer.length());

        CompressedDataBuffer result = new CompressedDataBuffer(pointer, descriptor);

        return result;
    }
}
