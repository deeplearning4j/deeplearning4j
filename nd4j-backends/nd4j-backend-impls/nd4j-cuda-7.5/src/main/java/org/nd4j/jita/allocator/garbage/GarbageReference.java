package org.nd4j.jita.allocator.garbage;

import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * @author raver119@gmail.com
 */
public class GarbageReference extends WeakReference<BaseDataBuffer> {
    private final AllocationPoint point;

    public GarbageReference(BaseDataBuffer referent, ReferenceQueue<? super BaseDataBuffer> q, AllocationPoint point) {
        super(referent, q);
        this.point = point;
    }

    public AllocationPoint getPoint(){
        return point;
    }
}
