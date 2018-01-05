package org.nd4j.autodiff.util.cloner;

import com.rits.cloning.IDeepCloner;
import com.rits.cloning.IFastCloner;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class INDArrayFastCloner implements IFastCloner {
    @Override
    public Object clone(Object o, IDeepCloner iDeepCloner, Map<Object, Object> map) {
        return ((INDArray) o).dup();
    }
}
