package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 * f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
 * where shift = max_i(x_i)
 */
@EqualsAndHashCode
@Getter
public class ActivationSoftmax extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new OldSoftMax(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        /*
        //libnd4j only returns diagonal elements, fix in libnd4j?
        //derivative of softmax(in) shape = minibatchxclasses should give minibatch x classes x classes
        int miniBatchSize = in.shape()[0];
        int classSize = in.shape()[1];
        //if (in.rank() != 2) throw exception?
        INDArray z = Nd4j.zeros(miniBatchSize,classSize,classSize);
        INDArray i = Nd4j.eye(classSize);
        INDArray out = z.dup();
        
        //identity matrix extended to 3d
        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(z,i,out,new int[] {1,2}));
        
        //D_jS_j = S_i * (delta_ij - S_j)
        Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(out,in,z,new int[] {0,1}));//1-p or -p
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(z,in,out,new int[] {0,1}));//p*(1-p) or -pi*pj
        
        gradient = out;
        */
        //use loss fn utils and push this for next release
        //        Nd4j.getExecutioner().execAndReturn(new SoftMax(in).derivative());
        //        return in;

        INDArray out = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(in));

        INDArray x = out.mul(epsilon).sum(1);
        INDArray dLdz = out.mul(epsilon.subColumnVector(x));

        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "softmax";
    }

}
