package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 14/09/2016.
 */
public class LossUtil {

    public static INDArray dLdZsoftmaxPreOut(INDArray dlda, INDArray z){
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
     * @return
     */
    public static INDArray dLdZsoftmax(INDArray dlda, INDArray a){
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
     * @return
     */
    public static INDArray dLdZsoftmaxi(INDArray dlda, INDArray a){
        INDArray x = a.mul(dlda).sum(1);
        return a.muli(dlda.subiColumnVector(x));
    }
}
