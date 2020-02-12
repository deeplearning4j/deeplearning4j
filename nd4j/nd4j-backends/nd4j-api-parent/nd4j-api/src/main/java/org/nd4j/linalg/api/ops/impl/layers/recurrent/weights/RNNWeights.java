package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import java.lang.reflect.Array;
import java.util.Arrays;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public abstract class RNNWeights {
    public abstract SDVariable[] args();

    public abstract INDArray[] ndarrayArgs();



    protected static <T> T[] filterNonNull(T... in){
        int count = 0;
        for( int i=0; i<in.length; i++ ) {
            if (in[i] != null) count++;
        }
        T[] out = (T[]) Array.newInstance(in.getClass().getComponentType(), count);
        int j=0;
        for( int i=0; i<in.length; i++ ){
            if(in[i] != null){
                out[j++] = in[i];
            }
        }
        return out;
    }

    public SDVariable[] argsWithInputs(SDVariable... inputs){
        return ArrayUtil.combine(inputs, args());
    }

    public INDArray[] argsWithInputs(INDArray... inputs) { return  ArrayUtil.combine(inputs, ndarrayArgs()); };


}
