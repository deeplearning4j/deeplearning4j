package org.nd4j.linalg.jcublas.context.pool.factory;

import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Creates new streams
 *
 * @author Adam Gibson
 */
public class StreamItemFactory extends BasePooledObjectFactory<CUstream> {
    @Override
    public CUstream create() throws Exception {
        CUstream stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
        return stream;
    }

    @Override
    public PooledObject<CUstream> wrap(CUstream cUstream) {
        return new DefaultPooledObject<>(cUstream);
    }

    @Override
    public void destroyObject(PooledObject<CUstream> p) throws Exception {
        super.destroyObject(p);
        JCudaDriver.cuStreamDestroy(p.getObject());
    }
}
