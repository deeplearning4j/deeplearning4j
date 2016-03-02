package org.nd4j.linalg.jcublas.context.pool.factory;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Creates old streams
 *
 * @author Adam Gibson
 */
public class OldStreamItemFactory extends BasePooledObjectFactory<cudaStream_t> {


    @Override
    public cudaStream_t create() throws Exception {
        try {
            cudaStream_t oldStream = new cudaStream_t();
            JCuda.cudaStreamCreate(oldStream);
            return oldStream;
        }catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public PooledObject<cudaStream_t> wrap(cudaStream_t cudaStream_t) {
        return new DefaultPooledObject<>(cudaStream_t);
    }

    @Override
    public void destroyObject(PooledObject<cudaStream_t> p) throws Exception {
        super.destroyObject(p);
        JCuda.cudaStreamDestroy(p.getObject());
    }
}
