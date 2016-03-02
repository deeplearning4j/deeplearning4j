package org.nd4j.linalg.jcublas.context.pool;

import jcuda.runtime.cudaStream_t;
import org.apache.commons.pool2.BaseObjectPool;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.AbandonedConfig;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;

/**
 * Generates streams for the old api
 *
 * @author Adam Gibson
 */
public class OldStreamPool extends GenericObjectPool<cudaStream_t> {
    public OldStreamPool(PooledObjectFactory<cudaStream_t> factory) {
        super(factory);
    }

    public OldStreamPool(PooledObjectFactory<cudaStream_t> factory, GenericObjectPoolConfig config) {
        super(factory, config);
    }

    public OldStreamPool(PooledObjectFactory<cudaStream_t> factory, GenericObjectPoolConfig config, AbandonedConfig abandonedConfig) {
        super(factory, config, abandonedConfig);
    }




}
