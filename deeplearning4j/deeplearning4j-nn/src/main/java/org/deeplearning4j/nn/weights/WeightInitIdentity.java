package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * Weights are set to an identity matrix. Note: can only be used when nIn==nOut.
 * For Dense layers, this means square weight matrix
 * For convolution layers, an additional constraint is that kernel size must be odd length in all dimensions.
 * the we
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitIdentity implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        if (shape[0] != shape[1]) {
            throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                    + Arrays.toString(shape) + ": weights must be a square matrix for identity");
        }
        switch (shape.length) {
            case 2:
               return setIdentity2D(shape, order, paramView);
            case 3:
            case 4:
            case 5:
                return setIdentityConv(shape, order, paramView);
                default: throw new IllegalStateException("Identity mapping for " + shape.length +" dimensions not defined!");
        }
    }

    private INDArray setIdentity2D(long[] shape, char order, INDArray paramView) {
        INDArray ret;
        if (order == Nd4j.order()) {
            ret = Nd4j.eye(shape[0]);
        } else {
            ret = Nd4j.createUninitialized(shape, order).assign(Nd4j.eye(shape[0]));
        }
        INDArray flat = Nd4j.toFlattened(order, ret);
        paramView.assign(flat);
        return paramView.reshape(order, shape);
    }

    /**
     * Set identity mapping for convolution layers. When viewed as an NxM matrix of kernel tensors,
     * identity mapping is when parameters is a diagonal matrix of identity kernels.
     * @param shape Shape of parameters
     * @param order Order of parameters
     * @param paramView View of parameters
     * @return A reshaped view of paramView which results in identity mapping when used in convolution layers
     */
    private INDArray setIdentityConv(long[] shape, char order, INDArray paramView) {
        final INDArrayIndex[] indArrayIndices = new INDArrayIndex[shape.length];
        for(int i = 2; i < shape.length; i++) {
            if(shape[i] % 2 == 0) {
                throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                        + Arrays.toString(shape) + "! Must have odd sized kernels!");
            }
            indArrayIndices[i] = NDArrayIndex.point(shape[i] / 2);
        }

        paramView.assign(Nd4j.zeros(paramView.shape()));
        final INDArray params =paramView.reshape(order, shape);
        for (int i = 0; i < shape[0]; i++) {
            indArrayIndices[0] = NDArrayIndex.point(i);
            indArrayIndices[1] = NDArrayIndex.point(i);
            params.put(indArrayIndices, Nd4j.ones(1));
        }
        return params;
    }
}
