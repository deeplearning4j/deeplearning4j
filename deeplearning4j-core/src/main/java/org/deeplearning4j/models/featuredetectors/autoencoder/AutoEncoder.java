package org.deeplearning4j.models.featuredetectors.autoencoder;



import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.berkeley.Pair;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;


/**
 * Normal 2 layer back propagation network
 * @author Adam Gibson
 */
public class AutoEncoder extends BasePretrainNetwork {


    public AutoEncoder(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
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
    public Gradient getGradient() {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        //feed forward
        INDArray out = transform(input);

        INDArray diff = input.sub(out);

        INDArray wGradient = diff.transpose().mmul(W);
        INDArray hBiasGradient = wGradient.sum(1);
        INDArray vBiasGradient = Nd4j.zeros(vBias.rows(), vBias.columns());

        Gradient ret =  createGradient(wGradient, vBiasGradient, hBiasGradient);
        return ret;

    }



    // Encode
    public INDArray getHiddenValues(INDArray x) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

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
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

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





}
