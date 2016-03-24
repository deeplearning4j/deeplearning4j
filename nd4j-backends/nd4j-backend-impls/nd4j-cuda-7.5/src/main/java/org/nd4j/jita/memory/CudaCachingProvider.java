package org.nd4j.jita.memory;

import jcuda.Pointer;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;



/**
 * @author raver119@gmail.com
 */
public class CudaCachingProvider extends CudaDirectProvider implements MemoryProvider {
    private static Logger log = LoggerFactory.getLogger(CudaCachingProvider.class);

    private ConcurrentHashMap<AllocationShape, Queue<Pointer>> zeroCache = new ConcurrentHashMap<>();

    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        if (location == AllocationStatus.HOST) {
            Queue<Pointer> queue = zeroCache.get(shape);
            if (queue != null) {
                log.info("YAY! We have something in cache!");
                throw new UnsupportedOperationException("Not implemented yet");

            } else return super.malloc(shape, point, location);
        } else return super.malloc(shape, point, location);
    }

    @Override
    public void free(AllocationPoint point) {
        if (point.getAllocationStatus() == AllocationStatus.DEVICE) {
            super.free(point);
        } else {
            super.free(point);
        }
    }
}
