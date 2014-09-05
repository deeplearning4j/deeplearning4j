package org.deeplearning4j.optimize.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;

/**
 * Output layer optimizer
 * @author Adam Gibson
 */
public class OutputLayerOptimizer implements OptimizableByGradientValueMatrix {

    private OutputLayer logReg;
    private double lr;
    private int currIteration = -1;



    public OutputLayerOptimizer(OutputLayer logReg, double lr) {
        super();
        this.logReg = logReg;
        this.lr = lr;
    }

    @Override
    public void setCurrentIteration(int value) {
        this.currIteration = value;
    }

    @Override
    public int getNumParameters() {
        return logReg.getW().length() + logReg.getB().length();

    }


    public void getParameters(float[] buffer) {
        for(int i = 0; i < buffer.length; i++) {
            buffer[i] = getParameter(i);
        }




    }


    public float getParameter(int index) {
        if(index >= logReg.getW().length())
            return (float) logReg.getB().getScalar(index - logReg.getW().length()).element();
        return (float) logReg.getW().getScalar(index).element();
    }


    public void setParameters(float[] params) {
        for(int i = 0; i < params.length; i++) {
            setParameter(i,params[i]);
        }
    }

    @Override
    public void setParameter(int index, float value) {
        if(index >= logReg.getW().length())
            logReg.getB().putScalar(index - logReg.getW().length(), value);
        else
            logReg.getW().putScalar(index, value);
    }


    public void getValueGradient(float[] buffer) {
        OutputLayerGradient grad = logReg.getGradient(lr);
        for(int i = 0; i < buffer.length; i++) {
            if(i < logReg.getW().length())
                buffer[i] = (float) grad.getwGradient().getScalar(i).element();
            else
                buffer[i] = (float) grad.getbGradient().getScalar(i - logReg.getW().length()).element();

        }
    }

    @Override
    public float getValue() {
        return -logReg.score();
    }

    @Override
    public INDArray getParameters() {
      return logReg.params();
    }

    @Override
    public void setParameters(INDArray params) {
        if(logReg.conf().isConstrainGradientToUnitNorm())
            params.divi(params.normmax(Integer.MAX_VALUE));
        logReg.setParams(params);
    }

    @Override
    public INDArray getValueGradient(int currIteration) {
        this.currIteration = currIteration;
        OutputLayerGradient grad = logReg.getGradient(lr);
        if(logReg.getW().length() != grad.getwGradient().length())
            throw new IllegalStateException("Illegal length for gradient");
        if(logReg.getB().length() != grad.getbGradient().length())
            throw new IllegalStateException("Illegal length for gradient");


        return Nd4j.toFlattened(grad.getwGradient(),grad.getbGradient());
    }



}
