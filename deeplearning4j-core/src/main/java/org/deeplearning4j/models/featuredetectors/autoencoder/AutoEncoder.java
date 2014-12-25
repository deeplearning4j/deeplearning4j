package org.deeplearning4j.models.featuredetectors.autoencoder;



import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.berkeley.Pair;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.BaseNeuralNetwork;

import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;

/**
 * Normal 2 layer back propagation network
 * @author Adam Gibson
 */
public class AutoEncoder extends BaseNeuralNetwork {


    private AutoEncoder(){}
    public AutoEncoder(INDArray input, INDArray W, INDArray hbias, INDArray vbias,NeuralNetConfiguration conf) {
        super(input, W, hbias, vbias,conf);
    }
   /**
     * All neural networks are based on this idea of
     * minimizing reconstruction error.
     * Both RBMs and Denoising AutoEncoders
     * have a component for reconstructing, ala different implementations.
     *
     * @param x the input to transform
     * @return the reconstructed input
     */
    @Override
    public INDArray transform(INDArray x) {
        return getReconstructedInput(x);
    }


    @Override
    public INDArray hiddenActivation(INDArray input) {
        return getHiddenValues(input);

    }

    /**
     * iterate one iteration of the network
     *
     * @param input  the input to iterate on
     */
    @Override
    public void iterate(INDArray input) {
        NeuralNetworkGradient gradient = getGradient();
        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());
    }

    @Override
    public NeuralNetworkGradient getGradient() {
        double lr = conf.getLr();
        int iterations = conf.getNumIterations();


        //feed forward
        INDArray out = transform(input);

        INDArray diff = input.sub(out);

        INDArray wGradient = diff.transpose().mmul(W);
        INDArray hBiasGradient = wGradient.sum(1);
        INDArray vBiasGradient = Nd4j.zeros(vBias.rows(), vBias.columns());

        NeuralNetworkGradient ret =  new NeuralNetworkGradient(wGradient,vBiasGradient,hBiasGradient);
        return ret;

    }



    // Encode
    public INDArray getHiddenValues(INDArray x) {
        INDArray preAct;
        if(conf.isConcatBiases()) {
            INDArray concat = Nd4j.vstack(W,hBias.transpose());
            preAct =  x.mmul(concat);

        }
        else
            preAct = x.mmul(W).addiRowVector(hBias);
        INDArray ret = Transforms.sigmoid(preAct);
        applyDropOutIfNecessary(ret);
        return ret;
    }

    // Decode
    public INDArray getReconstructedInput(INDArray y) {
        if(conf.isConcatBiases()) {
            //row already accounted for earlier
            INDArray preAct = y.mmul(W.transpose());
            preAct = Nd4j.hstack(preAct,Nd4j.ones(preAct.rows(),1));
            return Transforms.sigmoid(preAct);
        }
        else {
            INDArray preAct = y.mmul(W.transpose());
            preAct.addiRowVector(vBias);
            return Transforms.sigmoid(preAct);
        }

    }



    /**
     * Sample hidden mean and sample
     * given visible
     *
     * @param v the  the visible input
     * @return a pair with mean, sample
     */
    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(INDArray v) {
        INDArray out = transform(v);
        return new Pair<>(out,out);
    }

    /**
     * Sample visible mean and sample
     * given hidden
     *
     * @param h the  the hidden input
     * @return a pair with mean, sample
     */
    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray out = transform(h);
        return new Pair<>(out,out);
    }



    public static class Builder extends BaseNeuralNetwork.Builder<AutoEncoder> {

        public Builder() {
            this.clazz = AutoEncoder.class;
        }




        @Override
        public AutoEncoder build() {
            AutoEncoder ret = super.build();
            return ret;
        }



    }



}
