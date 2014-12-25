package org.deeplearning4j.models.featuredetectors.da;

import static org.deeplearning4j.util.MathUtils.binomial;

import java.io.Serializable;


import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;


/**
 * Denoising Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoder extends BaseNeuralNetwork implements Serializable  {


    private static final long serialVersionUID = -6445530486350763837L;

    private DenoisingAutoEncoder() {}


    public DenoisingAutoEncoder(INDArray input, INDArray W, INDArray hbias, INDArray vbias,NeuralNetConfiguration conf) {
        super(input, W, hbias, vbias,conf);
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


    @Override
    public INDArray hiddenActivation(INDArray input) {
        return getHiddenValues(input);
    }

    // Encode
    public INDArray getHiddenValues(INDArray x) {
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
     * Perform one iteration of training
     * @param x the input
      */
    public void train(INDArray x) {
        if(x != null)
            this.input = x;
        this.lastMiniBatchSize = x.rows();
        NeuralNetworkGradient gradient = getGradient();
        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());

    }

    @Override
    public INDArray transform(INDArray x) {
        INDArray y = getHiddenValues(x);
        return getReconstructedInput(y);
    }



    public static class Builder extends BaseNeuralNetwork.Builder<DenoisingAutoEncoder> {
        public Builder()  {
            this.clazz = DenoisingAutoEncoder.class;
        }



        @Override
        public Builder withClazz(Class<? extends BaseNeuralNetwork> clazz) {
            super.withClazz(clazz);
            return this;
        }


        @Override
        public Builder withInput(INDArray input) {
            super.withInput(input);
            return this;
        }



        @Override
        public Builder withWeights(INDArray W) {
            super.withWeights(W);
            return this;
        }

        @Override
        public Builder withVisibleBias(INDArray vBias) {
            super.withVisibleBias(vBias);
            return this;
        }

        @Override
        public Builder withHBias(INDArray hBias) {
            super.withHBias(hBias);
            return this;
        }



    }








    @Override
    public void iterate(INDArray input ) {
        if(input != null )
            this.input = preProcessInput(input);
        this.lastMiniBatchSize = input.rows();
        NeuralNetworkGradient gradient = getGradient();

        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());
    }



    @Override
    public void iterationDone(int iteration) {
        int plotEpochs = conf.getRenderWeightIterations();
        if(plotEpochs <= 0)
            return;
        if(iteration % plotEpochs == 0 || iteration == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(),getInput().rows());
        }
    }

    @Override
    public  NeuralNetworkGradient getGradient() {

        double corruptionLevel = conf.getCorruptionLevel();
        double lr = conf.getLr();
        int iteration = conf.getNumIterations();

        if(wAdaGrad != null)
            this.wAdaGrad.setMasterStepSize(lr);
        if(hBiasAdaGrad != null )
            this.hBiasAdaGrad.setMasterStepSize(lr);
        if(vBiasAdaGrad != null)
            vBiasAdaGrad.setMasterStepSize(lr);


        INDArray corruptedX = getCorruptedInput(input, corruptionLevel);
        INDArray y = getHiddenValues(corruptedX);

        INDArray z = getReconstructedInput(y);
        INDArray visibleLoss =  input.sub(z);
        INDArray hiddenLoss = conf.getSparsity() == 0 ? visibleLoss.mmul(W).mul(y).mul(y.rsub(1)) :
        	visibleLoss.mmul(W).mul(y).mul(y.add(- conf.getSparsity()));


        INDArray wGradient = corruptedX.transpose().mmul(hiddenLoss).add(visibleLoss.transpose().mmul(y));

        INDArray hBiasGradient = hiddenLoss.mean(0);
        INDArray vBiasGradient = visibleLoss.mean(0);

        NeuralNetworkGradient gradient = new NeuralNetworkGradient(wGradient,vBiasGradient,hBiasGradient);

        return gradient;
    }







}
