package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import java.util.Arrays;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public abstract class RNNWeights {
    public abstract SDVariable[] args();

    public abstract INDArray[] ndarrayArgs();

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

    protected static INDArray[] filterNonNull(INDArray... args){
        int count = 0;
        for(INDArray v : args){
            if(v != null){
                count++;
            }
        }

        INDArray[] res = new INDArray[count];

        int i = 0;

        for(INDArray v : args){
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

    public INDArray[] argsWithInputs(INDArray... inputs) { return  ArrayUtil.combine(inputs, ndarrayArgs()); };


}
