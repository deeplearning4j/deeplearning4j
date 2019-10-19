package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.*;

@Slf4j
public class CloseValidationMemoryMgr implements SessionMemMgr {

    private final SessionMemMgr underlying;
    private final Map<INDArray, Boolean> released = new IdentityHashMap<>();

    public CloseValidationMemoryMgr(SessionMemMgr underlying){
        this.underlying = underlying;
    }

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        INDArray out = underlying.allocate(detached, dataType, shape);
        released.put(out, false);
        return out;
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        INDArray out = underlying.allocate(detached, descriptor);
        released.put(out, false);
        return out;
    }

    @Override
    public void release(INDArray array) {
        Preconditions.checkState(released.containsKey(array), "Attempting to release an array that was not allocated by" +
                " this memory manager: id=%s", array.getId());
        Preconditions.checkState(!released.get(array), "Attempting to release an array that was already deallocated by" +
                " an earlier release call to this memory manager: id=%s", array.getId());
        released.put(array, true);
    }

    @Override
    public void close() {
        underlying.close();
    }

    public void assertAllReleasedExcept(@NonNull Collection<INDArray> except){
        for(INDArray arr : except){
            if(!released.containsKey(arr)){
                throw new IllegalStateException("Array " + arr.getId() + " was not originally allocated by the memory manager");
            }

            boolean released = this.released.get(arr);
            if(released){
                throw new IllegalStateException("Specified output array (id=" + arr.getId() + ") should not have been deallocated but was");
            }
        }

        Set<INDArray> exceptSet = Collections.newSetFromMap(new IdentityHashMap<INDArray, Boolean>());
        exceptSet.addAll(except);

        int numNotClosed = 0;
        Set<INDArray> notReleased = Collections.newSetFromMap(new IdentityHashMap<INDArray, Boolean>());
        for(Map.Entry<INDArray, Boolean> e : released.entrySet()){
            INDArray a = e.getKey();
            if(!exceptSet.contains(a)){
                boolean b = e.getValue();
                if(!b){
                    notReleased.add(a);
                    numNotClosed++;
                    log.info("Not released: array id {}", a.getId());
                }
            }
        }

        if (numNotClosed > 0) {
            throw new IllegalStateException(numNotClosed + " arrays were not released but should have been");
        }
    }
}
