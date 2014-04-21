package org.deeplearning4j.rbm;

import static org.deeplearning4j.util.MatrixUtil.*;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.Tensor;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.util.Convolution;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Convolutional RBM
 * @author Adam Gibson
 */
public class ConvolutionalRBM extends RBM  {

    /**
     *
     */
    private static final long serialVersionUID = 6868729665328916878L;
    private int numFilters = 4;
    //top down signal from hidden feature maps to visibles
    private Tensor visI;
    //bottom up signal from visibles to hiddens
    private Tensor hidI;
    private Tensor W;
    private int[] stride = {2,2};
    protected int[] visibleSize;
    protected int[] filterSize;
    protected int[] fmSize;
    private static Logger log = LoggerFactory.getLogger(ConvolutionalRBM.class);
    protected boolean convolutionInitCalled = false;


    protected ConvolutionalRBM() {}




    protected ConvolutionalRBM(DoubleMatrix input, int nVisible, int n_hidden, DoubleMatrix W,
                               DoubleMatrix hbias, DoubleMatrix vBias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        super(input, nVisible, n_hidden, W, hbias, vBias, rng,fanIn,dist);
    }



    /* convolutional specific init */
    private void convolutionInit() {
        if(convolutionInitCalled)
            return;
        W = new Tensor(filterSize[0],filterSize[1],numFilters);
        visI = Tensor.zeros(visibleSize[0],visibleSize[1],numFilters);
        hidI = Tensor.zeros(fmSize[0],fmSize[1],numFilters);
        convolutionInitCalled = true;
        vBias = DoubleMatrix.zeros(1);
        hBias = DoubleMatrix.zeros(numFilters);
    }



    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     *
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    @Override
    public Tensor propUp(DoubleMatrix v) {
        for(int i = 0; i < numFilters; i++) {
            DoubleMatrix reversedSlice =  MatrixUtil.reverse(W.getSlice(i));
            DoubleMatrix slice =  Convolution.conv2d(v, reversedSlice, Convolution.Type.VALID).add(hBias.get(i));
            hidI.setSlice(i,slice);

        }

        Tensor expHidI = hidI.exp();

        Tensor eHid = expHidI.div(pool(expHidI).add(1));
        return eHid;
    }

    /**
     * Calculates the activation of the hidden:
     * sigmoid(h * W + vbias)
     *
     * @param h the hidden layer
     * @return the approximated output of the hidden layer
     */
    @Override
    public Tensor propDown(DoubleMatrix h) {
        Tensor h1 = (Tensor) h;
        for(int i = 0; i < numFilters; i++) {
            /*
               Each tensor only has one slice, need to figure out what's going on here
             */
            DoubleMatrix conv = Convolution.conv2d
                    (h1.getSlice(i), W.getSlice(i),
                            Convolution.Type.FULL);
            log.info("Slice shape " + conv.rows  + " with " + conv.columns);
            visI.setSlice(i,conv);
        }


        return  new Tensor(sigmoid(visI.sliceElementSums().add(vBias)));
    }





    public Tensor pool(Tensor input) {
        int nCols = input.columns();
        int nRows = input.rows;
        int yStride = stride[0];
        int xStride = stride[1];

        Tensor ret = Tensor.zeros(input.rows,input.columns,input.slices());
        int endRowBlock =  (int) Math.ceil(nRows / yStride);
        for(int i = 1;i < endRowBlock; i++) {
            int rowsMin = (i -1)  * yStride + 1;
            int rowsMax = i * yStride;
            int endColBlock = (int) Math.ceil(nCols / xStride);
            for(int j = 1; j < endColBlock; j++) {
                int cols = (j - 1)  * xStride + 1;
                int colsMax = j  * xStride;
                double blockVal = input.columnsSums().sum();
                int rowLength = rowsMax - rowsMin;
                int colLength = colsMax - cols;
                DoubleMatrix block = new DoubleMatrix(rowLength,colLength);
                MatrixUtil.assign(block,blockVal);
                ret.put(RangeUtils.interval(rowsMin,rowsMax),RangeUtils.interval(cols,colsMax),block);

            }

        }
        return ret;
    }

    @Override
    public DoubleMatrix getW() {
        return W;
    }

    /**
     * Reconstruction entropy.
     * This compares the similarity of two probability
     * distributions, in this case that would be the input
     * and the reconstructed input with gaussian noise.
     * This will account for either regularization or none
     * depending on the configuration.
     *
     * @return reconstruction error
     */
    @Override
    public double getReConstructionCrossEntropy() {
        return negativeLogLikelihood();
    }

    /**
     * Guess the visible values given the hidden
     * @param h
     * @return
     */
    @Override
    public Pair<DoubleMatrix,DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
        Tensor v1Mean = propDown(h);
        Tensor v1Sample = new Tensor(binomial(v1Mean, 1, rng));
        return new Pair<>((DoubleMatrix)v1Mean,(DoubleMatrix) v1Sample);
    }


    /**
     * Binomial sampling of the hidden values given visible
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
        Tensor h1Mean = propUp(v);
        Tensor h1Sample = new Tensor(binomial(h1Mean, 1, rng));
        //apply dropout
        applyDropOutIfNecessary(h1Sample);
        return new Pair<>((DoubleMatrix)h1Mean,(DoubleMatrix) h1Sample);

    }

    @Override
    public NeuralNetworkGradient getGradient(Object[] params) {
        int k = (int) params[0];
        double learningRate = (double) params[1];


        if(wAdaGrad != null)
            wAdaGrad.setMasterStepSize(learningRate);
        if(hBiasAdaGrad != null )
            hBiasAdaGrad.setMasterStepSize(learningRate);
        if(vBiasAdaGrad != null)
            vBiasAdaGrad.setMasterStepSize(learningRate);

		/*
		 * Cost and updates dictionary.
		 * This is the update rules for weights and biases
		 */
		/*
		 * Start the gibbs sampling.
		 */
        Tensor chainStart = this.propUp(input);



		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 *
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values
		 * or averages.
		 *
		 */

        Tensor nvSamples = null;
        Tensor hiddenMeans = chainStart;
        for(int i = 0; i < k; i++) {
            nvSamples = new Tensor(MatrixUtil.binomial(propDown(hiddenMeans),1,rng));
            hiddenMeans = propUp(nvSamples);
        }

		/*
		 * Update gradient parameters
		 */

        Tensor wGradient = new Tensor(W.rows(),W.columns(),W.slices());
        for(int i = 0; i < numFilters; i++) {
            wGradient.setSlice(i,Convolution.conv2d(input,
                    chainStart.getSlice(i), Convolution.Type.VALID)
                    .sub(Convolution.conv2d(nvSamples,
                            MatrixUtil.reverse(hiddenMeans.getSlice(i)),
                            Convolution.Type.VALID)));
        }




        DoubleMatrix hBiasGradient = null;

        //update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
        hBiasGradient = DoubleMatrix.scalar(chainStart.sub(hiddenMeans).columnSums().sum());




        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        DoubleMatrix  vBiasGradient = DoubleMatrix.scalar((input.sub(nvSamples)).columnSums().sum());
        NeuralNetworkGradient ret = new NeuralNetworkGradient(wGradient, vBiasGradient, hBiasGradient);

        updateGradientAccordingToParams(ret, learningRate);
        triggerGradientEvents(ret);

        return ret;
    }


    public static class Builder extends BaseNeuralNetwork.Builder<ConvolutionalRBM> {

        protected int numFilters = 4;
        protected int[] stride = {2,2};
        protected int[] visibleSize;
        protected int[] filterSize;
        protected int[] fmSize;


        public Builder() {
            this.clazz = ConvolutionalRBM.class;

        }

        public Builder withFilterSize(int[] filterSize) {
            if(filterSize == null || filterSize.length != 2)
                throw new IllegalArgumentException("Filter size must be of length 2");
            this.filterSize = filterSize;
            return this;
        }


        public Builder withVisibleSize(int[] visibleSize) {
            if(visibleSize == null || visibleSize.length != 2)
                throw new IllegalArgumentException("Visible size must be of length 2");
            this.visibleSize = visibleSize;
            return this;
        }

        public Builder withStride(int[] stride) {
            this.stride = stride;
            return this;
        }

        public Builder withNumFilters(int numFilters) {
            this.numFilters = numFilters;
            return this;
        }



        public ConvolutionalRBM build() {
            ConvolutionalRBM ret = (ConvolutionalRBM) super.build();
            if(filterSize == null)
                throw new IllegalStateException("Please specify a filter size");
            if(visibleSize == null)
                throw new IllegalStateException("Please specify a viisble size");
            ret.numFilters = numFilters;
            ret.stride = stride;
            fmSize = new int[2];
            fmSize[0] = visibleSize[0] - filterSize[0] + 1;
            fmSize[1] = visibleSize[1] - filterSize[1] + 1;
            ret.fmSize = fmSize;
            ret.visibleSize = visibleSize;
            ret.filterSize = filterSize;
            ret.convolutionInit();
            return ret;

        }



    }


}
