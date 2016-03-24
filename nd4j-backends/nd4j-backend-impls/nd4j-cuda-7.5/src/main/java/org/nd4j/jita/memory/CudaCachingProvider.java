package org.nd4j.jita.memory;

import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.pointers.PointersPair;

/**
 * @author raver119@gmail.com
 */
public class CudaCachingProvider extends CudaDirectProvider implements MemoryProvider {

    @Override
    public PointersPair malloc(AllocationShape shape, AllocationPoint point, AllocationStatus location) {
        return null;
    }

    @Override
    public void free(AllocationPoint point) {

    }
}
