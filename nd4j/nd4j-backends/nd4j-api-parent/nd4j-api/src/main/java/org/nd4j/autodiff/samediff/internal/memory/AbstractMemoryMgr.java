package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Abstract memory manager, that implements ulike and dup methods using the underlying allocate methods
 *
 * @author Alex Black
 */
public abstract class AbstractMemoryMgr implements SessionMemMgr {

    @Override
    public INDArray ulike(@NonNull INDArray arr) {
        return allocate(false, arr.dataType(), arr.shape());
    }

    @Override
    public INDArray dup(@NonNull INDArray arr) {
        INDArray out = ulike(arr);
        out.assign(arr);
        return out;
    }
}
