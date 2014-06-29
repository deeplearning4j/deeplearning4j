package org.deeplearning4j.da;

import static org.deeplearning4j.util.MathUtils.binomial;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;

import java.io.Serializable;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.sda.DenoisingAutoEncoderOptimizer;
import org.jblas.DoubleMatrix;


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


    private DenoisingAutoEncoder(DoubleMatrix input, int nVisible, int nHidden,
                                 DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
                                 RandomGenerator rng,double fanIn,RealDistribution dist) {
        super(input, nVisible, nHidden, W, hbias, vbias, rng,fanIn,dist);
    }

    /**
     * Corrupts the given input by doing a binomial sampling
     * given the corruption level
     * @param x the input to corrupt
     * @param corruptionLevel the corruption value
     * @return the binomial sampled corrupted input
     */
    public DoubleMatrix getCorruptedInput(DoubleMatrix x, double corruptionLevel) {
        DoubleMatrix tilde_x = DoubleMatrix.zeros(x.rows,x.columns);
        for(int i = 0; i < x.rows; i++)
            for(int j = 0; j < x.columns; j++)
                tilde_x.put(i,j,binomial(rng,1,1 - corruptionLevel));
        DoubleMatrix  ret = tilde_x.mul(x);
        return ret;
    }




    @Override
    public Pair<DoubleMatrix, DoubleMatrix> sampleHiddenGivenVisible(
            DoubleMatrix v) {
        DoubleMatrix ret = getHiddenValues(v);
        return new Pair<>(ret,ret);
    }


    @Override
    public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(
            DoubleMatrix h) {
        DoubleMatrix ret = getReconstructedInput(h);
        return new Pair<>(ret,ret);
    }




    // Encode
    public DoubleMatrix getHiddenValues(DoubleMatrix x) {
        DoubleMatrix preAct;
        if(concatBiases) {
            DoubleMatrix concat = DoubleMatrix.concatVertically(W,hBias.transpose());
            preAct =  x.mmul(concat);

        }
        else
            preAct = x.mmul(W).addiRowVector(hBias);
        DoubleMatrix ret = sigmoid(preAct);
        applyDropOutIfNecessary(ret);
        return ret;
    }

    // Decode
    public DoubleMatrix getReconstructedInput(DoubleMatrix y) {
        if(concatBiases) {
            //row already accounted for earlier
            DoubleMatrix preAct = y.mmul(W.transpose());
            preAct = DoubleMatrix.concatHorizontally(preAct,DoubleMatrix.ones(preAct.rows,1));
            return sigmoid(preAct);
        }
        else {
            DoubleMatrix preAct = y.mmul(W.transpose());
            preAct.addiRowVector(vBias);
            return sigmoid(preAct);
        }

    }


    /**
     * Run a network optimizer
     * @param x the input
     * @param lr the learning rate
     * @param corruptionLevel the corruption level
     * @param iterations to run
     */
    public void trainTillConvergence(DoubleMatrix x, double lr,double corruptionLevel,int iterations) {
        if(x != null)
            input = preProcessInput(x);
        this.lastMiniBatchSize = x.rows;
        optimizer = new DenoisingAutoEncoderOptimizer(this,lr,new Object[]{corruptionLevel,lr,iterations}, optimizationAlgo, lossFunction);
        optimizer.train(x);
    }

    /**
     * Perform one iteration of training
     * @param x the input
     * @param lr the learning rate
     * @param corruptionLevel the corruption level to train with
     * @param iteration the current iteration
     */
    public void train(DoubleMatrix x,double lr,double corruptionLevel,int iteration) {
        if(x != null && cacheInput)
            this.input = x;
        this.lastMiniBatchSize = x.rows;
        NeuralNetworkGradient gradient = getGradient(new Object[]{corruptionLevel,lr,iteration});
        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());

    }

    @Override
    public DoubleMatrix reconstruct(DoubleMatrix x) {
        DoubleMatrix y = getHiddenValues(x);
        return getReconstructedInput(y);
    }



    public static class Builder extends BaseNeuralNetwork.Builder<DenoisingAutoEncoder> {
        public Builder()  {
            this.clazz = DenoisingAutoEncoder.class;
        }


        /**
         * Weight initialization scheme
         * @param weightInit the weight initialization scheme
         * @return
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Constrains gradient to unit norm when updating parameters
         *
         * @param constrainGradientToUnitNorm whether to constrain the gradient to unit norm or not
         * @return
         */
        @Override
        public Builder constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            super.constrainGradientToUnitNorm(constrainGradientToUnitNorm);
            return this;
        }

        @Override
        public Builder concatBiases(boolean concatBiases) {
            super.concatBiases(concatBiases);
            return this;
        }

        @Override
        public Builder  cacheInput(boolean cacheInput) {
            super.cacheInput(cacheInput);
            return this;
        }

        @Override
        public Builder applySparsity(boolean applySparsity) {
            super.applySparsity(applySparsity);
            return this;
        }

        @Override
        public Builder withOptmizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            super.withOptmizationAlgo(optimizationAlgo);
            return this;
        }

        @Override
        public Builder withLossFunction(LossFunction lossFunction) {
            super.withLossFunction(lossFunction);
            return this;
        }

        @Override
        public Builder withDropOut(double dropOut) {
            super.withDropOut(dropOut);
            return this;
        }

        @Override
        public Builder normalizeByInputRows(boolean normalizeByInputRows) {
            super.normalizeByInputRows(normalizeByInputRows);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        public Builder withDistribution(RealDistribution dist) {
            super.withDistribution(dist);
            return this;
        }

        @Override
        public Builder useRegularization(boolean useRegularization) {
            super.useRegularization(useRegularization);
            return this;
        }

        @Override
        public Builder fanIn(double fanIn) {
            super.fanIn(fanIn);
            return this;
        }

        @Override
        public Builder withL2(double l2) {
            super.withL2(l2);
            return this;
        }

        @Override
        public Builder renderWeights(int numEpochs) {
            super.renderWeights(numEpochs);
            return this;
        }


        @Override
        public Builder withClazz(Class<? extends BaseNeuralNetwork> clazz) {
            super.withClazz(clazz);
            return this;
        }

        @Override
        public Builder withSparsity(double sparsity) {
            super.withSparsity(sparsity);
            return this;
        }

        @Override
        public Builder withMomentum(double momentum) {
            super.withMomentum(momentum);
            return this;
        }

        @Override
        public Builder withInput(DoubleMatrix input) {
            super.withInput(input);
            return this;
        }



        @Override
        public Builder withWeights(DoubleMatrix W) {
            super.withWeights(W);
            return this;
        }

        @Override
        public Builder withVisibleBias(DoubleMatrix vBias) {
            super.withVisibleBias(vBias);
            return this;
        }

        @Override
        public Builder withHBias(DoubleMatrix hBias) {
            super.withHBias(hBias);
            return this;
        }

        @Override
        public Builder numberOfVisible(int numVisible) {
            super.numberOfVisible(numVisible);
            return this;
        }

        @Override
        public Builder numHidden(int numHidden) {
            super.numHidden(numHidden);
            return this;
        }

        @Override
        public Builder withRandom(RandomGenerator gen) {
            super.withRandom(gen);
            return this;
        }

    }



    @Override
    public void trainTillConvergence(DoubleMatrix input,double lr,Object[] params) {
        if(input != null && cacheInput)
            this.input = input;
        this.lastMiniBatchSize = input.rows;
        optimizer = new DenoisingAutoEncoderOptimizer(this,lr, params, optimizationAlgo, lossFunction);
        optimizer.train(input);
    }




    @Override
    public double lossFunction(Object[] params) {
        return negativeLogLikelihood();
    }




    @Override
    public void train(DoubleMatrix input,double lr,Object[] params) {
        double corruptionLevel = (double) params[0];
        if(input != null && cacheInput)
            this.input = preProcessInput(input);
        this.lastMiniBatchSize = input.rows;
        NeuralNetworkGradient gradient = getGradient(new Object[]{corruptionLevel,lr,0});

        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());
    }



    @Override
    public void iterationDone(int epoch) {
        int plotEpochs = getRenderIterations();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0 || epoch == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(new Object[]{0.3,0.001,1000}),getInput().rows);
        }
    }

    @Override
    public  NeuralNetworkGradient getGradient(Object[] params) {

        double corruptionLevel = (double) params[0];
        double lr = (double) params[1];
        int iteration = (int) params[2];

        if(wAdaGrad != null)
            this.wAdaGrad.setMasterStepSize(lr);
        if(hBiasAdaGrad != null )
            this.hBiasAdaGrad.setMasterStepSize(lr);
        if(vBiasAdaGrad != null)
            vBiasAdaGrad.setMasterStepSize(lr);


        DoubleMatrix corruptedX = getCorruptedInput(input, corruptionLevel);
        DoubleMatrix y = getHiddenValues(corruptedX);

        DoubleMatrix z = getReconstructedInput(y);
        DoubleMatrix visibleLoss =  input.sub(z);
        DoubleMatrix hiddenLoss = sparsity == 0 ? visibleLoss.mmul(W).mul(y).mul(oneMinus(y)) : visibleLoss.mmul(W).mul(y).mul(y.add(- sparsity));


        DoubleMatrix wGradient = corruptedX.transpose().mmul(hiddenLoss).add(visibleLoss.transpose().mmul(y));

        DoubleMatrix hBiasGradient = hiddenLoss.columnMeans();
        DoubleMatrix vBiasGradient = visibleLoss.columnMeans();

        NeuralNetworkGradient gradient = new NeuralNetworkGradient(wGradient,vBiasGradient,hBiasGradient);
        updateGradientAccordingToParams(gradient,iteration, lr);

        return gradient;
    }







}
