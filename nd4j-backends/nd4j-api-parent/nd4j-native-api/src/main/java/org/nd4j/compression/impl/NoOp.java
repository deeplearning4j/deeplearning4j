package org.nd4j.compression.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.compression.CompressionType;

/**
 * Dummy NoOp compressor, that actually does no compression.
 *
 * @author raver119@gmail.com
 */
public class NoOp extends AbstractCompressor {
    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "NOOP";
    }

    /**
     * This method returns compression type provided by specific NDArrayCompressor implementation
     *
     * @return
     */
    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSLESS;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {
        return buffer.dup();
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        return buffer.dup();
    }
}
