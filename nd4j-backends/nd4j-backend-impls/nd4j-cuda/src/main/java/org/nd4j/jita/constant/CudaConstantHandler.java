package org.nd4j.jita.constant;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ConstantHandler implementation for CUDA backend.
 *
 * @author raver119@gmail.com
 */
public class CudaConstantHandler extends BasicConstantHandler {
    private static Logger logger = LoggerFactory.getLogger(CudaConstantHandler.class);

    protected static final ConstantHandler wrappedHandler = ProtectedCudaConstantHandler.getInstance();

    public CudaConstantHandler() {

    }

    @Override
    public long moveToConstantSpace(DataBuffer dataBuffer) {
        return wrappedHandler.moveToConstantSpace(dataBuffer);
    }

    @Override
    public DataBuffer getConstantBuffer(int[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(float[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        return wrappedHandler.relocateConstantSpace(dataBuffer);
    }

    /**
     * This method removes all cached constants
     */
    @Override
    public void purgeConstants() {
        wrappedHandler.purgeConstants();
    }

    @Override
    public long getCachedBytes() {
        return wrappedHandler.getCachedBytes();
    }
}
