package org.deeplearning4j.nn.conf;

/**
 * @author raver119@gmail.com
 */
public enum CacheMode {
    /**
     * Device memory will be used for cache (if current backend support such differentiation)
     */
    DEVICE,

    /**
     * Host memory will be used for cache
     */
    HOST,

    /**
     * Cache won't be used during training
     */
    NONE
}
