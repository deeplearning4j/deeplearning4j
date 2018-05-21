package org.nd4j.autodiff.util.cloner;

import com.rits.cloning.IDeepCloner;
import com.rits.cloning.IFastCloner;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.util.Map;

public class DataBufferFastCloner implements IFastCloner {
    @Override
    public Object clone(Object o, IDeepCloner iDeepCloner, Map<Object, Object> map) {
        return ((DataBuffer)o).dup();
    }
}
