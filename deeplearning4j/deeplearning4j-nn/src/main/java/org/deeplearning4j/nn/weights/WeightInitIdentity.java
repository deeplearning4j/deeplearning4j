package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * : Weights are set to an identity matrix. Note: can only be used with square weight matrices
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitIdentity implements IWeightInit {

    @Override
    public INDArray init(long fanIn, long fanOut, long[] shape, char order, INDArray paramView) {
        if(shape.length != 2 || shape[0] != shape[1]){
            throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                    + Arrays.toString(shape) + ": weights must be a square matrix for identity");
        }
        INDArray ret;
        if(order == Nd4j.order()){
            ret = Nd4j.eye(shape[0]);
        } else {
            ret = Nd4j.createUninitialized(shape, order).assign(Nd4j.eye(shape[0]));
        }
        INDArray flat = Nd4j.toFlattened(order, ret);
        paramView.assign(flat);
        return paramView.reshape(order, shape);
    }
}
