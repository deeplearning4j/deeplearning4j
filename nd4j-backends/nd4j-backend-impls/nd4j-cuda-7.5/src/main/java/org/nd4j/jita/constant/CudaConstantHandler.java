package org.nd4j.jita.constant;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.utils.AllocationUtils;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.flow.FlowController;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

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
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        return wrappedHandler.relocateConstantSpace(dataBuffer);
    }
}
