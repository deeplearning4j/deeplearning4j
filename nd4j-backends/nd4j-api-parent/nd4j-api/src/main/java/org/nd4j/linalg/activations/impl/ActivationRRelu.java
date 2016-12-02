package org.nd4j.linalg.activations.impl;

//import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Arrays;

/**
 * Created by susaneraly on 12/1/16.
 */
public class ActivationRRelu implements IActivation {

    //How is the seed set for repeatability? This should be the seed from the conf.
    protected double l,u;
    protected boolean forwardSaved=false;
    private INDArray alpha; //save the alpha's for the backward pass

    public ActivationRRelu() {
        this(1/8.0,1/3.0);
    }

    public ActivationRRelu(double l, double u) {
        this.l = l;
        this.u = u;
    }

    public INDArray getAlpha() {
        if(forwardSaved) {
            return alpha;
        }
        else {
            //throw exception - alpha not saved from forward pass
        }
    }

    private void setAlpha(int[] inShape) {
        forwardSaved = true;
        alpha = Nd4j.rand(inShape, l, u, Nd4j.getRandom());
    }

    private boolean overWriteAlpha(int[] inShape) {
        return !forwardSaved && Arrays.equals(inShape,alpha.shape());

    }

    @Override
    public INDArray computeActivation(INDArray in) {
        /*
            https://github.com/torch/nn/blob/master/doc/transfer.md
            f = nn.RReLU([l, u[, inplace]])
            f(x) = max(0,x) + a * min(0, x)
            a ~ U(l, u)
         */
        if (overWriteAlpha(in.shape())) {
            setAlpha(in.shape());
        }
        INDArray activations = in.dup();
        BooleanIndexing.replaceWhere(activations, 0.0, Conditions.greaterThan(0.0));
        activations.muli(getAlpha());
        return activations.addi(Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in)));
    }

    public INDArray computeActivation(INDArray in, boolean training) {
        if (!training) {
            return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in.dup(), (l+u)/2));
        }
        else {
            return computeActivation(in);
        }
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        /*
            gradient
                1; x>=0
                alpha; x<0
         */
        if (overWriteAlpha(in)) {
            //should never have to overwrite alpha here, throw error/warning?
            //only as a result of user error
                //computegradient was called before compute activation
                //computegradient is called with a different shape than the shape compute activation was called with
        }
        INDArray gradients = in.dup();
        BooleanIndexing.replaceWhere(gradients, 1.0, Conditions.greaterThanOrEqual(0.0));
        BooleanIndexing.replaceWhere(gradients, getAlpha(), Conditions.lessThan(0));
        return gradients;
    }

    public INDArray computeGradient(INDArray in, boolean training) {
        if (!training) {
            //why would you ever need the gradient for test?
            return Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in.dup(), (l+u)/2).derivative());
        }
        else {
            return computeGradient(in);
        }
    }


    @Override
    public Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in) {
        if (overWriteAlpha(in.shape())) {
            setAlpha(in.shape());
        }
        return new Pair<INDArray, INDArray>(
                computeActivation(in),
                computeGradient(in)
        ); //thread safety?

    }

}
