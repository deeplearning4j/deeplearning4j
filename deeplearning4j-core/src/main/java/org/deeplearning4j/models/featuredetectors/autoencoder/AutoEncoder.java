package org.deeplearning4j.models.featuredetectors.autoencoder;

import static org.deeplearning4j.util.MathUtils.binomial;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;


/**
 *  Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 * @author Adam Gibson
 *
 */
public class AutoEncoder extends BasePretrainNetwork  {


    private static final long serialVersionUID = -6445530486350763837L;

    public AutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    public AutoEncoder(NeuralNetConfiguration conf, INDArray input) {
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
        INDArray corrupted = Nd4j.zeros(x.rows(), x.columns());
        for(int i = 0; i < x.rows(); i++)
            for(int j = 0; j < x.columns(); j++)
                corrupted.put(i,j,binomial(conf.getRng(),1,1 - corruptionLevel));
        corrupted.muli(x);
        return corrupted;
    }




    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(
            INDArray v) {
        INDArray ret = encode(v);
        return new Pair<>(ret,ret);
    }


    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(
            INDArray h) {
        INDArray ret = decode(h);
        return new Pair<>(ret,ret);
    }



    // Encode
    public INDArray encode(INDArray x) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        INDArray preAct;
        if(conf.isConcatBiases()) {
            INDArray concat = Nd4j.hstack(W,hBias.transpose());
            preAct =  x.mmul(concat);

        }
        else
            preAct = x.mmul(W).addiRowVector(hBias);
        INDArray ret = conf.getActivationFunction().apply(preAct);
        return ret;
    }

    // Decode
    public INDArray decode(INDArray y) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        if(conf.isConcatBiases()) {
            //row already accounted for earlier
            INDArray preAct = y.mmul(W.transpose());
            preAct = Nd4j.hstack(preAct,Nd4j.ones(preAct.rows(),1));
            return conf.getActivationFunction().apply(preAct);
        }
        else {
            INDArray preAct = y.mmul(W.transpose());
            preAct.addiRowVector(vBias);
            return conf.getActivationFunction().apply(preAct);
        }

    }






    @Override
    public INDArray transform(INDArray x) {
        INDArray y = encode(x);
        return decode(y);
    }





    @Override
    public  Gradient getGradient() {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);

        double corruptionLevel = conf.getCorruptionLevel();

        INDArray corruptedX = corruptionLevel > 0 ? getCorruptedInput(input, corruptionLevel) : input;
        INDArray y = encode(corruptedX);

        INDArray z = decode(y);
        INDArray visibleLoss =  input.sub(z);
        INDArray hiddenLoss = conf.getSparsity() == 0 ? visibleLoss.mmul(W).muli(y).muli(y.rsub(1)) :
                visibleLoss.mmul(W).muli(y).muli(y.addi(-conf.getSparsity()));


        INDArray wGradient = corruptedX.transpose().mmul(hiddenLoss).addi(visibleLoss.transpose().mmul(y));

        INDArray hBiasGradient = hiddenLoss.mean(0);
        INDArray vBiasGradient = visibleLoss.mean(0);

        Gradient gradient = createGradient(wGradient, vBiasGradient, hBiasGradient);

        return gradient;
    }
}
