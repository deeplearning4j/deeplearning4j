package org.nd4j.linalg.compression;

import lombok.Data;

/**
 * @author raver119@gmail.com
 */
@Data
public class CompressionDescriptor {
    private CompressionType compressionType;
    private String compressionAlgorithm;
    private long originalLength;
    private long compressedLength;
}
