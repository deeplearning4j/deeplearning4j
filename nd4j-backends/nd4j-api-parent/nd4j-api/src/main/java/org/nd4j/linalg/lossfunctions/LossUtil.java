package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by Alex on 14/09/2016.
 */
public class LossUtil {

    /**
     * @deprecated No longer used
     */
    @Deprecated
    public static INDArray dLdZsoftmaxPreOut(INDArray dlda, INDArray z) {
        return dLdZsoftmax(dlda, Nd4j.getExecutioner().execAndReturn(new SoftMax(z.dup())));
    }

    /**
     * Calculate dL/dz for softmax activation function, from dL/da and a, where<br>
     * a: output activations<br>
     * dL/da: derivative of loss function with respect to the output activations<br>
     * <b>Note</b>: This version does NOT modify either input arrays. This is less efficient. For a more efficient implementation
     * that modifies the input arrays, use {@link #dLdZsoftmaxi(INDArray, INDArray)}.
     *
     * @param dlda    derivative of loss function with respect to the output activations (shape [minibatchSize, nOut])
     * @param a       output activations array (shape [minibatchSize, nOut])
     * @deprecated No longer used
     */
    @Deprecated
    public static INDArray dLdZsoftmax(INDArray dlda, INDArray a) {
        INDArray x = a.mul(dlda).sum(1);
        return a.mul(dlda.subColumnVector(x));
    }

    /**
     * Calculate dL/dz for softmax activation function, from dL/da and a, where<br>
     * a: output activations<br>
     * dL/da: derivative of loss function with respect to the output activations<br>
     * <b>Note</b>: This version WILL modify both input arrays (for efficiency). If this is not acceptable, use
     * {@link #dLdZsoftmax(INDArray, INDArray)}.
     *
     * @param dlda    derivative of loss function with respect to the output activations (shape [minibatchSize, nOut])
     * @param a       output activations array (shape [minibatchSize, nOut])
     * @deprecated No longer used
     */
    @Deprecated
    public static INDArray dLdZsoftmaxi(INDArray dlda, INDArray a) {
        INDArray x = a.mul(dlda).sum(1);
        return a.muli(dlda.subiColumnVector(x));
    }

    public static boolean isPerOutputMasking(INDArray to, INDArray mask){
        return !mask.isColumnVector() ||  Arrays.equals(to.shape(), mask.shape());
    }

    public static void applyMask(INDArray to, INDArray mask){
        //Two possibilities exist: it's *per example* masking, or it's *per output* masking
        //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
        if(mask.isColumnVector()) {
            to.muliColumnVector(mask);
        } else if(Arrays.equals(to.shape(), mask.shape())){
            to.muli(mask);
        } else {
            throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                    + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                    + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
        }
    }
}
