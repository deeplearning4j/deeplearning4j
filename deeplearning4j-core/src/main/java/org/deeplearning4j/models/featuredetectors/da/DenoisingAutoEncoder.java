package org.deeplearning4j.models.featuredetectors.da;

import static org.deeplearning4j.util.MathUtils.binomial;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.BasePretrainNetwork;


/**
 * Denoising Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoder extends BasePretrainNetwork  {


    private static final long serialVersionUID = -6445530486350763837L;

    public DenoisingAutoEncoder(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    /**
     * Corrupts the given input by doing a binomial sampling
     * given the corruption level
     * @param x the input to corrupt
     * @param corruptionLevel the corruption value
     * @return the binomial sampled corrupted input
     */
    public INDArray getCorruptedInput(INDArray x, double corruptionLevel) {
        INDArray tilde_x = Nd4j.zeros(x.rows(), x.columns());
        for(int i = 0; i < x.rows(); i++)
            for(int j = 0; j < x.columns(); j++)
                tilde_x.put(i,j,binomial(conf.getRng(),1,1 - corruptionLevel));
        INDArray  ret = tilde_x.mul(x);
        return ret;
    }




    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(
            INDArray v) {
        INDArray ret = getHiddenValues(v);
        return new Pair<>(ret,ret);
    }


    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(
            INDArray h) {
        INDArray ret = getReconstructedInput(h);
        return new Pair<>(ret,ret);
    }



    // Encode
    public INDArray getHiddenValues(INDArray x) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        INDArray preAct;
        if(conf.isConcatBiases()) {
            INDArray concat = Nd4j.hstack(W,hBias.transpose());
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






    @Override
    public INDArray transform(INDArray x) {
        INDArray y = getHiddenValues(x);
        return getReconstructedInput(y);
    }





    @Override
    public  Gradient getGradient() {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);

        double corruptionLevel = conf.getCorruptionLevel();

        INDArray corruptedX = getCorruptedInput(input, corruptionLevel);
        INDArray y = getHiddenValues(corruptedX);

        INDArray z = getReconstructedInput(y);
        INDArray visibleLoss =  input.sub(z);
        INDArray hiddenLoss = conf.getSparsity() == 0 ? visibleLoss.mmul(W).mul(y).mul(y.rsub(1)) :
        	visibleLoss.mmul(W).mul(y).mul(y.add(- conf.getSparsity()));


        INDArray wGradient = corruptedX.transpose().mmul(hiddenLoss).add(visibleLoss.transpose().mmul(y));

        INDArray hBiasGradient = hiddenLoss.mean(0);
        INDArray vBiasGradient = visibleLoss.mean(0);

        Gradient gradient = createGradient(wGradient, vBiasGradient, hBiasGradient);

        return gradient;
    }







}
