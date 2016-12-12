package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.PReLU;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationPReLU implements IActivation {

    boolean initCalled = false;

    //this should come from the kind of layer it is
    //if it is a timeseries, it is equal to ??
    //if it is an image, it is equal to the number of channels
    //otherwise it is just a single param per layer (note not per perceptron - though the layer could override that potentially)
    private INDArray alpha;

    public ActivationPReLU(int np) {
        setNumParams(np);
    }

    @Override
    public INDArray computeActivation(INDArray in, boolean training) {
        computeActivation(in);
    }

    private INDArray computeActivation(INDArray in){
        //assert?
        // if minibatch - op over dimension
        // op over dimension is:
        //      if 3d (image) op over channel
        //      else a transform
        return Nd4j.getExecutioner().execAndReturn(new PReLU(in,alpha));
    }

    @Override
    public INDArray computeGradient(INDArray in) {
        return Nd4j.getExecutioner().execAndReturn(new LeakyReLUDerivative(in,alpha));
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
        return "prelu";
    }

    @Override
    public int numParams() {
        return alpha.equals(null)? 1: alpha.length();
    }

    @Override
    public void setParamsViewArray(INDArray paramsView) {
        if (!initCalled) {
            //view from params
            //randomly set alpha
            initCalled = true; //alpha is valid
        }
        //cannot overwrite
    }

    @Override
    public void setBackpropViewArray(INDArray in, INDArray params) {

    }
}
