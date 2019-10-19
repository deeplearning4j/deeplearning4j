package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

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
