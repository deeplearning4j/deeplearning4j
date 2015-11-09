package org.nd4j.linalg.jcublas.context.pool.factory;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;

/**
 * Created by agibsonccc on 10/8/15.
 */
public class CublasHandlePooledItemFactory extends BasePooledObjectFactory<cublasHandle> {
    @Override
    public cublasHandle create() throws Exception {
        cublasHandle  handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        return handle;
    }

    @Override
    public PooledObject<cublasHandle> wrap(cublasHandle cublasHandle) {
        return new DefaultPooledObject<>(cublasHandle);
    }

    @Override
    public void destroyObject(PooledObject<cublasHandle> p) throws Exception {
        super.destroyObject(p);
        JCublas2.cublasDestroy(p.getObject());
    }
}
