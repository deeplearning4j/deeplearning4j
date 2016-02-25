package org.nd4j.linalg.jcublas.context.pool;

import jcuda.jcublas.cublasHandle;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.AbandonedConfig;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;

/**
 * Created by agibsonccc on 10/8/15.
 */
public class CublasHandlePool extends GenericObjectPool<cublasHandle> {

    public CublasHandlePool(PooledObjectFactory<cublasHandle> factory) {
        super(factory);
    }

    public CublasHandlePool(PooledObjectFactory<cublasHandle> factory, GenericObjectPoolConfig config) {
        super(factory, config);
    }

    public CublasHandlePool(PooledObjectFactory<cublasHandle> factory, GenericObjectPoolConfig config, AbandonedConfig abandonedConfig) {
        super(factory, config, abandonedConfig);
    }


}
