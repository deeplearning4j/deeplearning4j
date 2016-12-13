package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationSoftmax implements IActivation{

    @Override
    public INDArray computeActivation(INDArray in, boolean training) {
        return computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        return Nd4j.getExecutioner().execAndReturn(new SoftMax(in));
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        //libnd4j returns an array in on the same size with each element equals p*(1-p)
        //needs to return an array with diagonal elements equal p*(1-p) and pi*pj elsewhere
        //fix in libnd4j?
        int classSize = in.shape()[1];
        int miniBatchSize = in.shape()[0];
        if (in.shape()[in.length()-1] == 2) {
            return Nd4j.getExecutioner().execAndReturn(new SoftMaxDerivative(in));
        }
        else {
            //if (in.rank() == 2) {
                INDArray z = Nd4j.zeros(miniBatchSize,classSize,classSize);
                INDArray i = Nd4j.eye(classSize);
                INDArray out = z.dup();

                //identity matrix extended to 3d
                Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(z,i,out,new int[] {1,2}));

                //D_jS_j = S_i * (delta_ij - S_j)
                Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(out,in,z,new int[] {0,1}));//reusing z
                //System.out.println("======== 1 - p ==============");
                //System.out.println(z);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(z,in,out,new int[] {0,1}));//reusing z
                //System.out.println("========My derivative==============");
                //System.out.println(out);
            //}
            //else {
                //throw exception??
            //}
                return out;
        }
    }

    @Override
    public Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in) {
        return new Pair<INDArray, INDArray>(
                computeActivation(in),
                computeGradient(in)
        );
    }

    @Override
    public String toString() {
        return "softmax";
    }

}
