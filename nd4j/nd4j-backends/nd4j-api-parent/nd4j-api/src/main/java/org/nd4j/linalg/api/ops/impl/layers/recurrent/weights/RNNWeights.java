package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import java.lang.reflect.Array;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.util.ArrayUtil;

public abstract class RNNWeights {
    public abstract SDVariable[] args();

    public abstract INDArray[] arrayArgs();

    protected static <T> T[] filterNonNull(T... args){
        int count = 0;
        for( int i=0; i<args.length; i++ ) {
            if (args[i] != null) count++;
        }
        T[] out = (T[]) Array.newInstance(args.getClass().getComponentType(), count);
        int j=0;
        for( int i=0; i<args.length; i++ ){
            if(args[i] != null){
                out[j++] = args[i];
            }
        }
        return out;
    }

    public SDVariable[] argsWithInputs(SDVariable... inputs){
        return ArrayUtil.combine(inputs, args());
    }

    public INDArray[] argsWithInputs(INDArray... inputs) {
        return ArrayUtil.combine(inputs, arrayArgs());
    }


}
