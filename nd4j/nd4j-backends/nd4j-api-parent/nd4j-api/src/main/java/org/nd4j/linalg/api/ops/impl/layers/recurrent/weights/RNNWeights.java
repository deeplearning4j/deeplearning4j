package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import java.util.Arrays;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public abstract class RNNWeights {
    public abstract SDVariable[] args();

    protected static SDVariable[] filterNonNull(SDVariable... args){
        int count = 0;
        for(SDVariable v : args){
            if(v != null){
                count++;
            }
        }

        SDVariable[] res = new SDVariable[count];

        int i = 0;

        for(SDVariable v : args){
            if(v != null){
                res[i] = v;
                i++;
            }
        }

        return res;
    }

    public SDVariable[] argsWithInputs(SDVariable... inputs){
        return ArrayUtil.combine(inputs, args());
    }

    public SDVariable[] argsWithInputs(INDArray... inputs) { return (SDVariable[]) ArrayUtil.combine(inputs, args()); };


}
