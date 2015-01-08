package org.deeplearning4j.models.featuredetectors.autoencoder.recursive;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.RecursiveParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 *
 *
 * @author Adam Gibson
 */
public class RecursiveAutoEncoder extends BaseLayer {

    public RecursiveAutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public double score() {
        return 0;
    }



    @Override
    public INDArray transform(INDArray data) {
        return conf.getActivationFunction().apply(data.mmul(params.get(RecursiveParamInitializer.W)).addiRowVector(params.get(RecursiveParamInitializer.BIAS)));
    }


    public INDArray decode(INDArray input) {
        return input.mmul(params.get(RecursiveParamInitializer.U).addiRowVector(params.get(RecursiveParamInitializer.C)));
    }


    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient getGradient() {
        Gradient gradient = new DefaultGradient();
        INDArray currInput = null;
        INDArray visibleLoss = null,hiddenLoss = null,cLoss = null,bLoss = null;
        /**
         * Going up the tree involves repeated calculations using the output of the previous autoencoder
         * for the next.
         * This starts with a base case at x[0] and x[1] and expands to subsequent layers.
         *
         * The error is the sum going up the tree.
         */
        for(int i = 0; i < input.rows(); i++) {
            INDArray combined = currInput == null ? Nd4j.concat(0,input.slice(i),input.slice(i + 1)) : Nd4j.concat(0,input.slice(i),currInput);
            //combine first 2: aka base case
            if(i == 0)
                i++;
            INDArray encoded = transform(combined);
            currInput = decode(encoded);

            INDArray currVisibleLoss = currInput.sub(combined);
            if(visibleLoss == null)
                visibleLoss = currVisibleLoss;
            else
                visibleLoss.addi(currVisibleLoss);

            INDArray currHiddenLoss = visibleLoss.mmul(getParam(RecursiveParamInitializer.W)).muli(encoded).muli(encoded.rsub(1));

            if(hiddenLoss == null)
                hiddenLoss = currHiddenLoss;
            else
                hiddenLoss.addi(currHiddenLoss);

            INDArray currCLoss = currVisibleLoss.mean(0);
            INDArray currBLoss = currHiddenLoss.mean(0);


            if(cLoss == null)
                cLoss = currCLoss;
            else
                cLoss.addi(currCLoss);
            if(bLoss == null)
                bLoss = currBLoss;
            else
                bLoss.addi(currBLoss);
        }
        return gradient;
    }



}
