package org.nd4j.linalg.compression;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface NDArrayCompressor {

    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     * @return
     */
    String getDescriptor();

    /**
     * This method returns compression type provided by specific NDArrayCompressor implementation
     * @return
     */
    CompressionType getCompressionType();

    INDArray compress(INDArray array);

    DataBuffer compress(DataBuffer buffer);

    INDArray decompress(INDArray array);

    DataBuffer decompress(DataBuffer buffer);


}
