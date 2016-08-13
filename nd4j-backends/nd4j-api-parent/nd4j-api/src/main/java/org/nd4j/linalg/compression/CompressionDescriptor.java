package org.nd4j.linalg.compression;

import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
@Data
public class CompressionDescriptor {
    private CompressionType compressionType;
    private String compressionAlgorithm;
    private long originalLength;
    private long compressedLength;
    private long numberOfElements;
    private long originalElementSize;

    public CompressionDescriptor() {

    }

    public CompressionDescriptor(DataBuffer buffer) {
        this.originalLength = buffer.length() * buffer.getElementSize();
        this.numberOfElements = buffer.length();
        this.originalElementSize = buffer.getElementSize();
    }

    public CompressionDescriptor(DataBuffer buffer, String algorithm) {
        this(buffer);
        this.compressionAlgorithm = algorithm;
    }

    public CompressionDescriptor(DataBuffer buffer, NDArrayCompressor compressor) {
        this(buffer, compressor.getDescriptor());
        this.compressionType = compressor.getCompressionType();
    }
}
