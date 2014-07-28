package org.deeplearning4j.rbm;

import static org.deeplearning4j.util.MatrixUtil.*;

import static org.deeplearning4j.util.Convolution.*;

import static org.deeplearning4j.util.Convolution.Type.*;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.linalg.FourDTensor;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.linalg.Tensor;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.nn.learning.FourDTensorAdaGrad;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.Convolution;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Convolutional RBM (binary)
 * @author Adam Gibson
 */
public class ConvolutionalRBM extends RBM  {

    /**
     *
     */
    private static final long serialVersionUID = 6868729665328916878L;
    //number of feature mapConvolution
    protected int[] numFilters = {4,4};
    //top down signal from hidden feature maps to visibles
    private FourDTensor visI;
    //bottom up signal from visibles to hiddens
    private FourDTensor hidI;
    //also called the filters
    private FourDTensor W;
    //overlapping pixels
    private int[] stride = {2,2};
    //visible layer size
    protected int[] visibleSize;
    protected int[] filterSize;
    //feature map sizes
    protected int[] fmSize;
    private static Logger log = LoggerFactory.getLogger(ConvolutionalRBM.class);
    protected boolean convolutionInitCalled = false;
    protected DoubleMatrix chainStart;
    //cache last propup/propdown
    protected FourDTensor eVis,eHid;
    protected DoubleMatrix wGradient,vBiasGradient,hBiasGradient;
    protected double sparseGain = 5;
    public int wRows = 0,wCols = 0,wSlices = 0;
    private FourDTensor featureMap;
    //same size as W
    protected FourDTensor dWeights;
    protected ConvolutionalRBM() {}




    protected ConvolutionalRBM(DoubleMatrix input, int nVisible, int n_hidden, DoubleMatrix W,
                               DoubleMatrix hbias, DoubleMatrix vBias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        super(input, nVisible, n_hidden, W, hbias, vBias, rng,fanIn,dist);
    }



    /* convolutional specific init */
    private void convolutionInit() {
        if(convolutionInitCalled)
            return;
        W = new FourDTensor(filterSize[0],filterSize[1],numFilters[0],numFilters[1]);
        wRows = W.rows();
        wCols = W.columns();
        wSlices = W.slices();
        visI = FourDTensor.zeros(visibleSize[0],visibleSize[1],numFilters[0],numFilters[1]);
        hidI = FourDTensor.zeros(fmSize[0],fmSize[1],numFilters[0],numFilters[1]);
        convolutionInitCalled = true;
        vBias = DoubleMatrix.zeros(1);
        hBias = DoubleMatrix.zeros(numFilters[0]);


        for(int i = 0; i < this.W.rows; i++)
            W.putRow(i,new DoubleMatrix(dist.sample(W.columns)));


        wAdaGrad = new FourDTensorAdaGrad(W.rows(),W.columns(),W.slices(),W.numTensors());
        vBiasAdaGrad = new AdaGrad(vBias.rows,vBias.columns);
        hBiasAdaGrad = new AdaGrad(hBias.rows,hBias.columns);
        convolutionInitCalled = true;
        dWeights = new FourDTensor(W.rows(),W.columns(),W.slices(),W.getNumTensor());
        //dont normalize by input rows here, the batch size can only be 1
        this.normalizeByInputRows = false;
    }



    /**
     * Calculates the activation of the visible :
     * sigmoid(v * W + hbias)
     *
     * @param v the visible layer
     * @return the approximated activations of the visible layer
     */
    @Override
    public FourDTensor propUp(DoubleMatrix v) {
        for(int i = 0; i < numFilters[0]; i++) {
            for(int j = 0; j < numFilters[1]; j++) {
                DoubleMatrix reversedSlice =  reverse(W.getSliceOfTensor(i,j));
                //a bias for each hidden unit
                DoubleMatrix slice = sigmoid(conv2d(v, reversedSlice, VALID).add(hBias.get(i)));
                hidI.put(i,j,slice);
            }


        }

        FourDTensor expHidI = MatrixUtil.exp(hidI);

        FourDTensor eHid = expHidI.div((DoubleMatrix) pool(expHidI).add(1));
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
    public FourDTensor propDown(DoubleMatrix h) {
        FourDTensor h1 = (FourDTensor) h;
        for(int i = 0; i < numFilters[0]; i++) {
            for(int j = 0; j < numFilters[1]; j++) {
 /*
               Each tensor only has one slice, need to figure out what's going on here
             */
                visI.put(j,i,sigmoid(conv2d(h1.getSlice(i), W.getSlice(i),FULL)));
            }

        }

        DoubleMatrix I = visI.sliceElementSums().add(vBias);
        if(visibleType == VisibleUnit.BINARY)
            I = sigmoid(I);

        FourDTensor ret =   new FourDTensor(I);
        this.eVis = ret;
        return ret;
    }


    /**
     * Pooled expectations given visibles for sampling
     * @param input the input to sample from
     * @return  the pooled expectations given visible
     */
    public Tensor poolGivenVis(DoubleMatrix input) {
        FourDTensor eHid = propUp(input);
        FourDTensor I = new FourDTensor(eHid.rows(),eHid.columns(),eHid.slices(),eHid.getNumTensor());
        for(int i = 0; i < W.slices(); i++) {
            for(int j = 0; j < W.getNumTensor(); j++) {
                I.setSlice(i,Convolution.conv2d(input,reverse(W.getSlice(i)), VALID).add(hBias.get(i)));
                I.put(j,i,Convolution.conv2d(input,reverse(W.getSlice(i)), VALID).add(hBias.get(i)));
            }
        }

        FourDTensor ret = FourDTensor.ones(I.rows(),I.columns(),I.slices(),I.numTensors());
        //1 / 1 + pool(exp(I))
        FourDTensor poolExpI = pool(MatrixUtil.exp(I)).add(1);
        FourDTensor sub = ret.div((DoubleMatrix) poolExpI);
        ret.subi(sub);
        return ret;
    }


    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param learningRate the learning rate for the current iteratiaon
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,double learningRate) {
        FourDTensor wGradient = (FourDTensor) gradient.getwGradient();

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
        if (applySparsity && this.hBiasGradient != null)
            applySparsity(hBiasGradient, learningRate);

        if (momentum != 0 && this.wGradient != null)
            wGradient.addi(this.wGradient.mul(momentum).add(wGradient.mul(1 - momentum)));


        if(momentum != 0 && this.vBiasGradient != null)
            vBiasGradient.addi(this.vBiasGradient.mul(momentum).add(vBiasGradient.mul(1 - momentum)));

        if(momentum != 0 && this.hBiasGradient != null)
            hBiasGradient.addi(this.hBiasGradient.mul(momentum).add(hBiasGradient.mul(1 - momentum)));


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
            wGradient.divi(lastMiniBatchSize);
            vBiasGradient.divi(lastMiniBatchSize);
            hBiasGradient.divi(lastMiniBatchSize);
        }

        this.wGradient = wGradient;
        this.vBiasGradient = vBiasGradient;
        this.hBiasGradient = hBiasGradient;

    }



    /**
     * Pooled expectations of I by summing over blocks of alpha
     * @param input the input to sum over
     * @return the pooled expectations
     */
    public FourDTensor pool(FourDTensor input) {
        int nCols = input.columns();
        int nRows = input.rows;
        int yStride = stride[0];
        int xStride = stride[1];

        FourDTensor ret = new FourDTensor(input.rows,input.columns,input.slices(),input.numTensors());
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
    public FourDTensor getW() {
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
        FourDTensor v1Mean = propDown(h);
        FourDTensor v1Sample = new FourDTensor(binomial(v1Mean, 1, rng));
        return new Pair<>((DoubleMatrix)v1Mean,(DoubleMatrix) v1Sample);
    }


    /**
     * Binomial sampling of the hidden values given visible
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {


        FourDTensor h1Mean = propUp(v);
        FourDTensor h1Sample = new FourDTensor(binomial(h1Mean, 1, rng));
        //apply dropout
        applyDropOutIfNecessary(h1Sample);
        return new Pair<>((DoubleMatrix)h1Mean,(DoubleMatrix) h1Sample);

    }


    /**
     * Backprop with the output being the reconstruction
     */
    @Override
    public void backProp(double lr,int iterations,Object[] extraParams) {
        boolean train = false;

        double currRecon = this.getReConstructionCrossEntropy();

        NeuralNetwork revert = clone();
        int numEpochs = 0;
        while(train) {
            if(numEpochs > iterations)
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

            int plotEpochs = getRenderIterations();
            if(plotEpochs > 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                if(numEpochs % plotEpochs == 0) {
                    plotter.plotNetworkGradient(this,getGradient(extraParams),getW().slices());
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
        FourDTensor chainStart = propUp(input);
        this.chainStart = chainStart;


		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 *
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values
		 * or averages.
		 *
		 */

        FourDTensor nvSamples = null;
        FourDTensor hiddenMeans = chainStart;
        //contrastive divergence
        for(int i = 0; i < k; i++) {
            nvSamples = propDown(binomial(eHid,1,rng));
            hiddenMeans = propUp(nvSamples);
        }

		/*
		 * Update gradient parameters
		 */

        FourDTensor wGradient = new FourDTensor(W.rows(),W.columns(),W.slices(),W.getNumTensor());
        for(int i = 0; i < numFilters[0]; i++)
            for(int j = 0; j < numFilters[1]; j++) {
                wGradient.put(j,i,conv2d(input,chainStart.getSliceOfTensor(j, i),VALID).sub(conv2d(nvSamples, reverse(hiddenMeans.getSliceOfTensor(j, i)), VALID)));
            }




        DoubleMatrix vBiasGradient = DoubleMatrix.scalar(chainStart.sub(hiddenMeans).columnSums().sum());

        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        DoubleMatrix  hBiasGradient = DoubleMatrix.scalar((input.sub(nvSamples)).columnSums().sum());
        NeuralNetworkGradient ret = new NeuralNetworkGradient(wGradient, vBiasGradient, hBiasGradient);


        updateGradientAccordingToParams(ret, learningRate);




        return ret;
    }

    /**
     * Applies sparsity to the passed in hbias gradient
     *
     * @param hBiasGradient the hbias gradient to apply to
     * @param learningRate  the learning rate used
     */
    @Override
    protected void applySparsity(DoubleMatrix hBiasGradient, double learningRate) {
        // dcSparse = self.lRate*self.sparseGain*(squeeze(self.sparsity -mean(mean(self.eHid0))));
        //self.c = self.c + dcSparse;
        if(sparsity != 0) {
            DoubleMatrix negMean = DoubleMatrix.scalar(sparseGain * (sparsity -chainStart.columnMeans().mean()));
            if(useAdaGrad)
                negMean.muli(hBiasAdaGrad.getLearningRates(hBiasGradient));
            else
                negMean.muli(learningRate);

            hBias.addi(negMean);



        }
    }

    public int[] getVisibleSize() {
        return visibleSize;
    }

    public void setVisibleSize(int[] visibleSize) {
        this.visibleSize = visibleSize;
    }

    public int[] getFilterSize() {
        return filterSize;
    }

    public void setFilterSize(int[] filterSize) {
        this.filterSize = filterSize;
    }

    public int[] getFmSize() {
        return fmSize;
    }

    public void setFmSize(int[] fmSize) {
        this.fmSize = fmSize;
    }

    public int[] getStride() {
        return stride;
    }

    public void setStride(int[] stride) {
        this.stride = stride;
    }

    public int[] getNumFilters() {

        return numFilters;
    }

    public void setNumFilters(int [] numFilters) {
        this.numFilters = numFilters;
    }

    public FourDTensor getVisI() {
        return visI;
    }

    public void setVisI(FourDTensor visI) {
        this.visI = visI;
    }

    public FourDTensor getHidI() {
        return hidI;
    }

    public void setHidI(FourDTensor hidI) {
        this.hidI = hidI;
    }

    public void setW(FourDTensor w) {
        W = w;
    }

    public boolean isConvolutionInitCalled() {

        return convolutionInitCalled;
    }

    public void setConvolutionInitCalled(boolean convolutionInitCalled) {
        this.convolutionInitCalled = convolutionInitCalled;
    }

    public DoubleMatrix getChainStart() {
        return chainStart;
    }

    public void setChainStart(DoubleMatrix chainStart) {
        this.chainStart = chainStart;
    }

    public Tensor geteVis() {
        return eVis;
    }

    public void seteVis(FourDTensor eVis) {
        this.eVis = eVis;
    }

    public Tensor geteHid() {
        return eHid;
    }

    public void seteHid(FourDTensor eHid) {
        this.eHid = eHid;
    }

    public DoubleMatrix getwGradient() {
        return wGradient;
    }

    public void setwGradient(DoubleMatrix wGradient) {
        this.wGradient = wGradient;
    }

    public DoubleMatrix getvBiasGradient() {
        return vBiasGradient;
    }

    public void setvBiasGradient(DoubleMatrix vBiasGradient) {
        this.vBiasGradient = vBiasGradient;
    }

    public DoubleMatrix gethBiasGradient() {
        return hBiasGradient;
    }

    public void sethBiasGradient(DoubleMatrix hBiasGradient) {
        this.hBiasGradient = hBiasGradient;
    }

    public double getSparseGain() {
        return sparseGain;
    }

    public void setSparseGain(double sparseGain) {
        this.sparseGain = sparseGain;
    }

    public int getwRows() {
        return wRows;
    }

    public void setwRows(int wRows) {
        this.wRows = wRows;
    }

    public int getwCols() {
        return wCols;
    }

    public void setwCols(int wCols) {
        this.wCols = wCols;
    }

    public int getwSlices() {
        return wSlices;
    }

    public void setwSlices(int wSlices) {
        this.wSlices = wSlices;
    }


    public FourDTensor getFeatureMap() {
        return featureMap;
    }

    public void setFeatureMap(FourDTensor featureMap) {
        this.featureMap = featureMap;
    }


    public FourDTensor getdWeights() {
        return dWeights;
    }

    public void setdWeights(FourDTensor dWeights) {
        this.dWeights = dWeights;
    }

    public static class Builder extends RBM.Builder {

        protected int[] numFilters = {4,4};
        protected int[] stride = {2,2};
        protected int[] visibleSize;
        protected int[] filterSize;
        protected int[] fmSize;
        protected double sparseGain = 5;

        public Builder() {
            this.clazz = ConvolutionalRBM.class;

        }

        @Override
        public Builder concatBiases(boolean concatBiases) {
            super.concatBiases(concatBiases);
            return this;
        }

        public Builder withFmSize(int[] fmSize) {
            this.fmSize = fmSize;
            return this;
        }


        public Builder withSparseGain(double sparseGain) {
            this.sparseGain = sparseGain;
            return this;
        }

        @Override
        public Builder withVisible(VisibleUnit visible) {
            super.withVisible(visible);
            return this;
        }

        @Override
        public Builder withHidden(HiddenUnit hidden) {
            super.withHidden(hidden);
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
        public ConvolutionalRBM buildEmpty() {
            return build();
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
        public Builder asType(Class<RBM> clazz) {
            super.asType(clazz);
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

        public Builder withNumFilters(int[] numFilters) {
            this.numFilters = numFilters;
            return this;
        }


        @Override
        public ConvolutionalRBM build() {
            ConvolutionalRBM ret = (ConvolutionalRBM) super.build();
            if(filterSize == null)
                throw new IllegalStateException("Please specify a filter size");
            if(visibleSize == null)
                throw new IllegalStateException("Please specify a visible size");
            ret.numFilters = numFilters;
            ret.stride = stride;
            ret.sparseGain = sparseGain;
            if(fmSize == null) {
                fmSize = new int[2];
                fmSize[0] = visibleSize[0] - filterSize[0] + 1;
                fmSize[1] = visibleSize[1] - filterSize[1] + 1;
            }

            ret.fmSize = fmSize;
            ret.visibleSize = visibleSize;
            ret.filterSize = filterSize;
            ret.convolutionInit();
            return ret;

        }



    }


}
