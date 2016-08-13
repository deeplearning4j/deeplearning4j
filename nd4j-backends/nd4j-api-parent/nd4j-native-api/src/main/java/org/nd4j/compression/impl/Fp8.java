package org.nd4j.compression.impl;

import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressionType;

/**
 * Compressor implementation based on 8 bitfloats, aka FP8 or Quarter
 * PLEASE NOTE: NOT IMPLEMENTED YET
 * @author raver119@gmail.com
 */
@Deprecated
public abstract class Fp8 extends AbstractCompressor {
    @Override
    public String getDescriptor() {
        return "FP8";
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
        return null;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        return null;
    }
}
