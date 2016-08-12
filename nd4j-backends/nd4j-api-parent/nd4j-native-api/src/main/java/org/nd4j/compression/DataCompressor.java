package org.nd4j.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class DataCompressor {
    private static final DataCompressor INSTANCE = new DataCompressor();

    private String defaultCompression = "FP16";

    private DataCompressor() {

    }

    public static DataCompressor getInstance() {
        return INSTANCE;
    }

    public void setDefaultCompression(@NonNull String descriptor) {
        synchronized (this) {
            defaultCompression = descriptor;
        }
    }

    public String getDefaultCompression() {
        synchronized (this) {
            return defaultCompression;
        }
    }

    public DataBuffer compress(DataBuffer buffer) {
        buffer.dataType();
        return buffer;
    }

    public INDArray compress(INDArray array) {
        return array;
    }
}
