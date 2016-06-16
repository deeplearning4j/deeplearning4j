package org.nd4j.linalg.jcublas;

import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.jita.constant.ConstantProtector;
import org.nd4j.jita.constant.ProtectedCudaShapeInfoProvider;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseShapeInfoProvider;
import org.nd4j.linalg.api.ndarray.ShapeInfoProvider;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.ShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CachedShapeInfoProvider extends BaseShapeInfoProvider {
    private static Logger logger = LoggerFactory.getLogger(CachedShapeInfoProvider.class);

    protected ShapeInfoProvider provider = ProtectedCudaShapeInfoProvider.getInstance();

    public CachedShapeInfoProvider() {

    }

    @Override
    // FIXME: this method should NOT be synchronized
    public synchronized DataBuffer createShapeInformation(int[] shape, int[] stride, int offset, int elementWiseStride, char order) {
        return provider.createShapeInformation(shape, stride, offset, elementWiseStride, order);
    }
}
