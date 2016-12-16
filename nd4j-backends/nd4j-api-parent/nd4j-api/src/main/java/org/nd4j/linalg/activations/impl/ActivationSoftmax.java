package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationSoftmax implements IActivation {

    @Override
    public void setActivation(INDArray in, INDArray activation, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new SoftMax(in,activation));
    }

    @Override
    public void setGradient(INDArray in, INDArray gradient) {
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
        Nd4j.getExecutioner().execAndReturn(new SoftMax(in,gradient).derivative());
    }

    @Override
    public void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient) {
        setActivation(in,activation, true);
        setGradient(in,gradient);
    }

}
