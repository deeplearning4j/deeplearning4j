package org.deeplearning4j.rbm;

import static org.deeplearning4j.util.MatrixUtil.*;

import static org.deeplearning4j.util.Convolution.*;

import static org.deeplearning4j.util.Convolution.Type.*;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.Tensor;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.Convolution;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
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
    //cache last propup/propdown
    protected Tensor eVis,eHid;

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
        vBias = DoubleMatrix.zeros(numFilters);
        hBias = DoubleMatrix.zeros(1);


        for(int i = 0; i < this.W.rows; i++)
            this.W.putRow(i,new DoubleMatrix(dist.sample(this.W.columns)));


        wAdaGrad = new AdaGrad(W.rows,W.columns);
        vBiasAdaGrad = new AdaGrad(vBias.rows,vBias.columns);
        hBiasAdaGrad = new AdaGrad(hBias.rows,hBias.columns);
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
            DoubleMatrix reversedSlice =  reverse(W.getSlice(i));
            DoubleMatrix slice =  conv2d(v, reversedSlice, VALID).add(vBias.get(0));
            hidI.setSlice(i,slice);

        }

        Tensor expHidI = hidI.exp();

        Tensor eHid = expHidI.div(pool(expHidI).add(1));
        this.eHid = eHid;
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
            DoubleMatrix conv = conv2d(h1.getSlice(i), W.getSlice(i),FULL);
            visI.setSlice(i,conv);
        }

        DoubleMatrix I = visI.sliceElementSums().add(hBias);
        I = sigmoid(I);

        Tensor ret =   new Tensor(I);
        this.eVis = ret;
        return ret;
    }


    /**
     * Pooled expectations given visibles for sampling
     * @param input the input to sample from
     * @return  the pooled expectations given visible
     */
    public Tensor poolGivenVis(DoubleMatrix input) {
        Tensor eHid = propUp(input);
        Tensor I = Tensor.zeros(eHid.rows(),eHid.columns(),eHid.slices());
        for(int i = 0; i < W.slices(); i++) {
            I.setSlice(i,Convolution.conv2d(input,reverse(W.getSlice(i)), VALID).add(hBias.get(i)));
        }

        Tensor ret = Tensor.ones(I.rows(),I.columns(),I.slices());
        //1 / 1 + pool(exp(I))
        Tensor poolExpI = pool(I.exp()).add(1);
        Tensor sub = ret.div(poolExpI);
        ret.subi(sub);
        return ret;
    }

    /**
     * Pooled expectations of I by summing over blocks of alpha
     * @param input the input to sum over
     * @return the pooled expectations
     */
    public Tensor pool(Tensor input) {
        int nCols = input.columns();
        int nRows = input.rows;
        int yStride = stride[0];
        int xStride = stride[1];

        Tensor ret = Tensor.zeros(input.rows,input.columns,input.slices());
        int endRowBlock =  (int) Math.ceil(nRows / yStride);
        for(int i = 1; i < endRowBlock; i++) {
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
                assign(block,blockVal);
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
        if(eVis == null)
            reconstruct(input);
        double squaredLoss = MatrixFunctions.pow(eVis.sub(input), 2).sum();
        return squaredLoss;
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


    /**
     * Backprop with the output being the reconstruction
     */
    @Override
    public void backProp(double lr,int epochs,Object[] extraParams) {
        boolean train = false;

        double currRecon = this.getReConstructionCrossEntropy();

        NeuralNetwork revert = clone();
        int numEpochs = 0;
        while(train) {
            if(numEpochs > epochs)
                break;

            NeuralNetworkGradient gradient = getGradient(extraParams);
            DoubleMatrix wLearningRates = getAdaGrad().getLearningRates(gradient.getwGradient());
            DoubleMatrix z = reconstruct(input);

            //Scale the input and reconstruction to see the relative difference in absolute space
           /*
           Other current problems: the hbias mmul output diff is being calculated wrong.
           We should be able to calculate the w gradient with 1 mmul.
            */
            DoubleMatrix scaledInput = input.dup();
            normalizeZeroMeanAndUnitVariance(scaledInput);
            normalizeZeroMeanAndUnitVariance(z);
            DoubleMatrix outputDiff = z.sub(scaledInput);
            //changes in error relative to neurons
            DoubleMatrix delta = W.mmul(outputDiff);
            //hidden activations
            DoubleMatrix hBiasMean = z.columnSums().transpose();

            if(isUseAdaGrad()) {
                delta.muli(wLearningRates);
            }
            else
                delta.muli(lr);

            if(momentum != 0)
                delta.muli(momentum).add(delta.mul(1 - momentum));

            if(normalizeByInputRows)
                delta.divi(input.rows);


            getW().addi(W.sub(delta));


            double newRecon = this.getReConstructionCrossEntropy();
            //prevent weights from exploding too far in either direction, we want this as close to zero as possible
            if(newRecon > currRecon || currRecon < 0 && newRecon < currRecon) {
                update((BaseNeuralNetwork) revert);
                log.info("Converged for new recon; breaking...");
                break;
            }

            else if(newRecon == currRecon)
                break;

            else {
                currRecon = newRecon;
                revert = clone();
                log.info("Recon went down " + currRecon);
            }

            numEpochs++;

            int plotEpochs = getRenderEpochs();
            if(plotEpochs > 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                if(numEpochs % plotEpochs == 0) {
                    plotter.plotNetworkGradient(this,getGradient(extraParams));
                }
            }

        }
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
        Tensor chainStart = propUp(input);



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
        //contrastive divergence
        for(int i = 0; i < k; i++) {
            nvSamples = propDown(binomial(eHid,1,rng));
            hiddenMeans = propUp(nvSamples);
        }

		/*
		 * Update gradient parameters
		 */

        Tensor wGradient = new Tensor(W.rows(),W.columns(),W.slices());
        for(int i = 0; i < numFilters; i++)
            wGradient.setSlice(i,conv2d(input,chainStart.getSlice(i),VALID).sub(conv2d(nvSamples,reverse(hiddenMeans.getSlice(i)),VALID)));





        DoubleMatrix vBiasGradient = DoubleMatrix.scalar(chainStart.sub(hiddenMeans).columnSums().sum());

        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        DoubleMatrix  hBiasGradient = DoubleMatrix.scalar((input.sub(nvSamples)).columnSums().sum());
        NeuralNetworkGradient ret = new NeuralNetworkGradient(wGradient, vBiasGradient, hBiasGradient);

        updateGradientAccordingToParams(ret, learningRate);
        triggerGradientEvents(ret);

        return ret;
    }


    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param learningRate the learning rate for the current iteratiaon
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,double learningRate) {
        DoubleMatrix wGradient = gradient.getwGradient();
        DoubleMatrix hBiasGradient = gradient.gethBiasGradient();
        DoubleMatrix vBiasGradient = gradient.getvBiasGradient();
        DoubleMatrix wLearningRates = wAdaGrad.getLearningRates(wGradient);
        if (useAdaGrad)
            wGradient.muli(wLearningRates);
        else
            wGradient.muli(learningRate);

        if (useAdaGrad)
            hBiasGradient = hBiasGradient.mul(hBiasAdaGrad.getLearningRates(hBiasGradient)).add(hBiasGradient.mul(momentum));
        else
            hBiasGradient = hBiasGradient.mul(learningRate).add(hBiasGradient.mul(momentum));


        if (useAdaGrad)
            vBiasGradient = vBiasGradient.mul(vBiasAdaGrad.getLearningRates(vBiasGradient)).add(vBiasGradient.mul(momentum));
        else
            vBiasGradient = vBiasGradient.mul(learningRate).add(vBiasGradient.mul(momentum));



        //only do this with binary hidden layers
        if (applySparsity)
            applySparsity(hBiasGradient, learningRate);

        if (momentum != 0) {
            DoubleMatrix change = wGradient.mul(momentum).add(wGradient.mul(1 - momentum));
            wGradient.addi(change);

        }

        if(useRegularization) {
            if(l2 > 0) {
                DoubleMatrix penalized = W.mul(l2);
                if(useAdaGrad)
                    penalized.muli(wAdaGrad.getLearningRates(wGradient));
                else
                    penalized.muli(learningRate);

                wGradient.subi(penalized);

            }

        }


        if (normalizeByInputRows) {
            wGradient.divi(input.rows);
            vBiasGradient.divi(input.rows);
            hBiasGradient.divi(input.rows);
        }

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
