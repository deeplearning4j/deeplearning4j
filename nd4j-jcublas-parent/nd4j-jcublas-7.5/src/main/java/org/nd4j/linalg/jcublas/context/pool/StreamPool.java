package org.nd4j.linalg.jcublas.context.pool;

import jcuda.driver.CUstream;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.AbandonedConfig;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;

/**
 * Created by agibsonccc on 10/8/15.
 */
public class StreamPool extends GenericObjectPool<CUstream> {
    public StreamPool(PooledObjectFactory<CUstream> factory) {
        super(factory);
    }

    public StreamPool(PooledObjectFactory<CUstream> factory, GenericObjectPoolConfig config) {
        super(factory, config);
    }

    public StreamPool(PooledObjectFactory<CUstream> factory, GenericObjectPoolConfig config, AbandonedConfig abandonedConfig) {
        super(factory, config, abandonedConfig);
    }

}
