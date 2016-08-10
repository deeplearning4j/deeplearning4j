package org.nd4j.compression;

import lombok.NonNull;

/**
 * @author raver119@gmail.com
 */
public class DataCompressor {
    private static final DataCompressor INSTANCE = new DataCompressor();

    private String defaultCompression = "FP16";

    private DataCompressor() {

    }

    public DataCompressor getInstance() {
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
}
