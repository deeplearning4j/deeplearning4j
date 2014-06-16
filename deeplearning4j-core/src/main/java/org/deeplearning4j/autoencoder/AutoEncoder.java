package org.deeplearning4j.autoencoder;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.jblas.DoubleMatrix;

/**
 * Normal 2 layer back propagation network
 * @author Adam Gibson
 */
public class AutoEncoder extends BaseNeuralNetwork {

    protected ActivationFunction act = Activations.sigmoid();

    private AutoEncoder(){}

    /**
     * @param nVisible the number of outbound nodes
     * @param nHidden  the number of nodes in the hidden layer
     * @param W        the weights for this vector, maybe null, if so this will
     *                 create a matrix with nHidden x nVisible dimensions.
     * @param hbias
     * @param vbias
     * @param rng      the rng, if not a seed of 1234 is used.
     * @param fanIn
     * @param dist
     */
    private AutoEncoder(int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng, double fanIn, RealDistribution dist) {
        super(nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
    }

    /**
     * @param input    the input examples
     * @param nVisible the number of outbound nodes
     * @param nHidden  the number of nodes in the hidden layer
     * @param W        the weights for this vector, maybe null, if so this will
     *                 create a matrix with nHidden x nVisible dimensions.
     * @param hbias
     * @param vbias
     * @param rng      the rng, if not a seed of 1234 is used.
     * @param fanIn
     * @param dist
     */
    private AutoEncoder(DoubleMatrix input, int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng, double fanIn, RealDistribution dist) {
        super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
    }

    /**
     * All neural networks are based on this idea of
     * minimizing reconstruction error.
     * Both RBMs and Denoising AutoEncoders
     * have a component for reconstructing, ala different implementations.
     *
     * @param x the input to reconstruct
     * @return the reconstructed input
     */
    @Override
    public DoubleMatrix reconstruct(DoubleMatrix x) {
        return act.apply(x.mmul(W).addRowVector(hBias));
    }

    /**
     * The loss function (cross entropy, reconstruction error,...)
     *
     * @param params
     * @return the loss function
     */
    @Override
    public double lossFunction(Object[] params) {
        return squaredLoss();
    }

    /**
     * train one iteration of the network
     *
     * @param input  the input to train on
     * @param lr     the learning rate to train at
     * @param params the extra params (k, corruption level,...)
     */
    @Override
    public void train(DoubleMatrix input, double lr, Object[] params) {
        NeuralNetworkGradient gradient = getGradient(new Object[]{lr});
        vBias.addi(gradient.getvBiasGradient());
        W.addi(gradient.getwGradient());
        hBias.addi(gradient.gethBiasGradient());
    }

    @Override
    public NeuralNetworkGradient getGradient(Object[] params) {
        double lr = (double) params[0];
        int iterations = (int) params[1];


        //feed forward
        DoubleMatrix out = reconstruct(input);

        DoubleMatrix diff = input.sub(out);

        DoubleMatrix backWard = diff.mul(input).mul(out).mul(out.neg().addi(1));

        DoubleMatrix wGradient = backWard.transpose().mmul(W);
        DoubleMatrix hBiasGradient = wGradient.columnMeans();
        DoubleMatrix vBiasGradient = DoubleMatrix.zeros(vBias.rows,vBias.columns);

        NeuralNetworkGradient ret =  new NeuralNetworkGradient(wGradient,vBiasGradient,hBiasGradient);
        updateGradientAccordingToParams(ret, iterations,lr);
         return ret;

    }

    /**
     * Sample hidden mean and sample
     * given visible
     *
     * @param v the  the visible input
     * @return a pair with mean, sample
     */
    @Override
    public Pair<DoubleMatrix, DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
        DoubleMatrix out = reconstruct(v);
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
    public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
        DoubleMatrix out = reconstruct(h);
        return new Pair<>(out,out);
    }

    /**
     * Trains via an optimization algorithm such as SGD or Conjugate Gradient
     *
     * @param input  the input to train on
     * @param lr     the learning rate to use
     * @param params the params (k,corruption level, max epochs,...)
     */
    @Override
    public void trainTillConvergence(DoubleMatrix input, double lr, Object[] params) {
        AutoEncoderOptimizer o = new AutoEncoderOptimizer(this,lr,params,optimizationAlgo,lossFunction);
        o.train(input);
    }


    public static class Builder extends BaseNeuralNetwork.Builder<AutoEncoder> {
        private ActivationFunction act = Activations.sigmoid();

        public Builder() {
            this.clazz = AutoEncoder.class;
        }

        public Builder withActivation(ActivationFunction act) {
            this.act = act;
            return this;
        }


        @Override
        public AutoEncoder build() {
            AutoEncoder ret = super.build();
            ret.act = this.act;
            return ret;
        }



    }



}
