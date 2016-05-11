package org.nd4j.jita.allocator.tad;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.ShapeDescriptor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class ConstantTADManager extends BasicTADManager {
    protected Map<TadDescriptor, DataBuffer> tadCache = new ConcurrentHashMap<>();

    @Override
    public DataBuffer getTADOnlyShapeInfo(INDArray array, int[] dimension, int dimensionLength) {
        /*
            so, we check, if we have things cached. If we don't - we just create new TAD shape, and push it to constant memory
        */

        TadDescriptor descriptor = new TadDescriptor(array, dimension, dimensionLength);
        if (!tadCache.containsKey(descriptor)) {
            DataBuffer buffer = super.getTADOnlyShapeInfo(array, dimension, dimensionLength);

            // so, at this point we have buffer valid on host side. And we just need to replace DevicePointer with constant pointer
            tadCache.put(descriptor, buffer);
        }

        return tadCache.get(descriptor);
    }
}
