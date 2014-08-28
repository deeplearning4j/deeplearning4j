package org.deeplearning4j.models.featuredetectors.da;

import static org.deeplearning4j.util.MathUtils.binomial;

import java.io.Serializable;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.optimize.optimizers.da.DenoisingAutoEncoderOptimizer;


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
    public INDArray getCorruptedInput(INDArray x, float corruptionLevel) {
        INDArray tilde_x = NDArrays.zeros(x.rows(), x.columns());
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
        INDArray preAct;
        if(conf.isConcatBiases()) {
            INDArray concat = NDArrays.concatVertically(W,hBias.transpose());
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
            preAct = NDArrays.concatHorizontally(preAct,NDArrays.ones(preAct.rows(),1));
            return Transforms.sigmoid(preAct);
        }
        else {
            INDArray preAct = y.mmul(W.transpose());
            preAct.addiRowVector(vBias);
            return Transforms.sigmoid(preAct);
        }

    }


    /**
     * Run a network optimizer
     * @param x the input
     * @param lr the learning rate
     * @param corruptionLevel the corruption level
     * @param iterations to run
     */
    public void trainTillConvergence(INDArray x, float lr,float corruptionLevel,int iterations) {
        if(x != null)
            input = preProcessInput(x);
        this.lastMiniBatchSize = x.rows();
        optimizer = new DenoisingAutoEncoderOptimizer(this,lr,new Object[]{corruptionLevel,lr,iterations}, conf.getOptimizationAlgo(), conf.getLossFunction());
        optimizer.train(x);
    }

    /**
     * Perform one iteration of training
     * @param x the input
     * @param lr the learning rate
     * @param corruptionLevel the corruption level to iterate with
     * @param iteration the current iteration
     */
    public void train(INDArray x,float lr,float corruptionLevel,int iteration) {
        if(x != null)
            this.input = x;
        this.lastMiniBatchSize = x.rows();
        NeuralNetworkGradient gradient = getGradient(new Object[]{corruptionLevel,lr,iteration});
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
    public void fit(INDArray input, Object[] params) {
        if(input != null)
            this.input = input;
        this.lastMiniBatchSize = input.rows();
        optimizer = new DenoisingAutoEncoderOptimizer(this,conf.getLr(), params, conf.getOptimizationAlgo(), conf.getLossFunction());
        optimizer.train(input);
    }




    @Override
    public float lossFunction(Object[] params) {
        return negativeLogLikelihood();
    }




    @Override
    public void iterate(INDArray input , Object[] params) {
        float corruptionLevel = (float) params[0];
        if(input != null )
            this.input = preProcessInput(input);
        this.lastMiniBatchSize = input.rows();
        NeuralNetworkGradient gradient = getGradient(new Object[]{corruptionLevel,conf.getLr(),0});

        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());
    }



    @Override
    public void iterationDone(int epoch) {
        int plotEpochs = conf.getRenderWeightsEveryNumEpochs();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0 || epoch == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(new Object[]{0.3,0.001,1000}),getInput().rows());
        }
    }

    @Override
    public  NeuralNetworkGradient getGradient(Object[] params) {

        float corruptionLevel = (float) params[0];
        float lr = (float) params[1];
        int iteration = (int) params[2];

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
        INDArray hiddenLoss = conf.getSparsity() == 0 ? visibleLoss.mmul(W).mul(y).mul(y.rsub(1)) : visibleLoss.mmul(W).mul(y).mul(y.add(- conf.getSparsity()));


        INDArray wGradient = corruptedX.transpose().mmul(hiddenLoss).add(visibleLoss.transpose().mmul(y));

        INDArray hBiasGradient = hiddenLoss.mean(1);
        INDArray vBiasGradient = visibleLoss.mean(1);

        NeuralNetworkGradient gradient = new NeuralNetworkGradient(wGradient,vBiasGradient,hBiasGradient);
        updateGradientAccordingToParams(gradient,iteration, lr);

        return gradient;
    }







}
