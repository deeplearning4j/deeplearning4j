package org.nd4j.jita.conf;

import lombok.NonNull;

/**
 * @author raver119@gmail.com
 */
public class CudaEnvironment {
    private static final CudaEnvironment INSTANCE = new CudaEnvironment();
    private volatile Configuration configuration;

    private CudaEnvironment() {

    }

    public static CudaEnvironment getInstance() {
        return INSTANCE;
    }

    public Configuration getConfiguration() {
        return configuration;
    }

    public void setConfiguration(@NonNull Configuration configuration) {
        this.configuration = configuration;
    }
}
