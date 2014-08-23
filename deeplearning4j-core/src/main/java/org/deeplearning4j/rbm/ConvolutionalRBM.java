package org.deeplearning4j.rbm;



import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.convolution.Convolution;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.plot.NeuralNetPlotter;
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
    private INDArray visI;
    //bottom up signal from visibles to hiddens
    private INDArray hidI;
    //also called the filters
    private INDArray W;
    //overlapping pixels
    private int[] stride = {2,2};
    //visible layer size
    protected int[] visibleSize;
    protected int[] filterSize;
    //feature map sizes
    protected int[] fmSize;
    private static Logger log = LoggerFactory.getLogger(ConvolutionalRBM.class);
    protected boolean convolutionInitCalled = false;
    protected INDArray chainStart;
    //cache last propup/propdown
    protected INDArray eVis,eHid;
    protected INDArray wGradient,vBiasGradient,hBiasGradient;
    protected double sparseGain = 5;
    public int wRows = 0,wCols = 0,wSlices = 0;
    private INDArray featureMap;
    //same size as W
    protected INDArray dWeights;
    protected ConvolutionalRBM() {}




    protected ConvolutionalRBM(INDArray input, int nVisible, int n_hidden, INDArray W,
                               INDArray hbias, INDArray vBias, RandomGenerator rng,double fanIn,RealDistribution dist) {
        super(input, nVisible, n_hidden, W, hbias, vBias, rng,fanIn,dist);
    }



    /* convolutional specific init */
    private void convolutionInit() {
        if(convolutionInitCalled)
            return;
        W = NDArrays.create(new int[]{filterSize[0], filterSize[1], numFilters[0], numFilters[1]});
        wRows = W.rows();
        wCols = W.columns();
        wSlices = W.slices();
        visI = NDArrays.zeros(new int[]{visibleSize[0],visibleSize[1],numFilters[0],numFilters[1]});
        hidI = NDArrays.zeros(new int[]{fmSize[0],fmSize[1],numFilters[0],numFilters[1]});
        convolutionInitCalled = true;
        vBias = NDArrays.zeros(1);
        hBias = NDArrays.zeros(numFilters[0]);


        for(int i = 0; i < this.W.rows(); i++)
            W.putRow(i,NDArrays.create(dist.sample(W.columns())));


        wAdaGrad = new AdaGrad(W.shape());
        vBiasAdaGrad = new AdaGrad(vBias.rows(),vBias.columns());
        hBiasAdaGrad = new AdaGrad(hBias.rows(),hBias.columns());
        convolutionInitCalled = true;
        dWeights = NDArrays.create(W.shape());
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
    public INDArray propUp(INDArray v) {
        for(int i = 0; i < numFilters[0]; i++) {
            for(int j = 0; j < numFilters[1]; j++) {
                INDArray reversedSlice =  NDArrays.reverse(W.slice(i).slice(j));
                //a bias for each hidden unit
                INDArray slice = Transforms.sigmoid(Convolution.convn(v, reversedSlice, Convolution.Type.VALID).addi(hBias.getScalar(i)));
                hidI.put(i,j,slice);
            }


        }

        INDArray expHidI = Transforms.exp(hidI.dup());

        INDArray eHid = expHidI.div(pool(expHidI).add(1));
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
    public INDArray propDown(INDArray h) {
        INDArray h1 =  h;
        for(int i = 0; i < numFilters[0]; i++) {
            for(int j = 0; j < numFilters[1]; j++) {

                visI.put(j,i,Transforms.sigmoid(Convolution.convn(h1.slice(i), W.slice(i), Convolution.Type.FULL)));
            }

        }

        INDArray I = visI.sum(visI.shape().length - 1).add(vBias);
        if(visibleType == VisibleUnit.BINARY)
            I = Transforms.sigmoid(I);


        this.eVis = I.dup();
        return this.eVis;
    }


    /**
     * Pooled expectations given visibles for sampling
     * @param input the input to sample from
     * @return  the pooled expectations given visible
     */
    public INDArray poolGivenVis(INDArray input) {
        INDArray eHid = propUp(input);
        INDArray I = NDArrays.create(eHid.shape());
        for(int i = 0; i < W.slices(); i++) {
            for(int j = 0; j < W.slices(); j++) {
                I.putSlice(i, Convolution.convn(input, NDArrays.reverse(W.slice(i)), Convolution.Type.VALID).addi(hBias.getScalar(i)));
                I.put(j,i,Convolution.convn(input, NDArrays.reverse(W.slice(i)), Convolution.Type.VALID).addi(hBias.getScalar(i)));
            }
        }

        INDArray ret = NDArrays.ones(I.shape());
        //1 / 1 + pool(exp(I))
        INDArray poolExpI = pool(Transforms.exp(I)).add(1);
        INDArray sub = ret.div( poolExpI);
        ret.subi(sub);
        return ret;
    }


    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param learningRate the learning rate for the current iteratiaon
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,double learningRate) {
        INDArray wGradient = gradient.getwGradient();

        INDArray hBiasGradient = gradient.gethBiasGradient();
        INDArray vBiasGradient = gradient.getvBiasGradient();
        INDArray wLearningRates = wAdaGrad.getLearningRates(wGradient);
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
                INDArray penalized = W.mul(l2);
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
    public INDArray pool(INDArray input) {
        int nCols = input.columns();
        int nRows = input.rows();
        int yStride = stride[0];
        int xStride = stride[1];

        INDArray ret = NDArrays.create(input.shape());
        int endRowBlock =  (int) Math.ceil(nRows / yStride);
        for(int i = 1; i < endRowBlock; i++) {
            int rowsMin = (i -1)  * yStride + 1;
            int rowsMax = i * yStride;
            int endColBlock = (int) Math.ceil(nCols / xStride);
            for(int j = 1; j < endColBlock; j++) {
                int cols = (j - 1)  * xStride + 1;
                int colsMax = j  * xStride;
                double blockVal = (double) input.sum(1).sum(Integer.MAX_VALUE).element();
                int rowLength = rowsMax - rowsMin;
                int colLength = colsMax - cols;
                INDArray block = NDArrays.create(rowLength,colLength);
                block.assign(blockVal);
                ret.put(new NDArrayIndex[]{NDArrayIndex.interval(rowsMin, rowsMax),NDArrayIndex.interval(cols,colsMax)},block);
            }

        }
        return ret;
    }

    @Override
    public INDArray getW() {
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
        double squaredLoss = (double) Transforms.pow(eVis.sub(input), 2).sum(Integer.MAX_VALUE).element();
        return squaredLoss;
    }

    /**
     * Guess the visible values given the hidden
     * @param h
     * @return
     */
    @Override
    public Pair<INDArray,INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray v1Mean = propDown(h);
        INDArray v1Sample = Sampling.binomial(v1Mean, 1, rng);
        return new Pair<>(v1Mean, v1Sample);
    }


    /**
     * Binomial sampling of the hidden values given visible
     * @param v the visible values
     * @return a binomial distribution containing the expected values and the samples
     */
    @Override
    public Pair<INDArray,INDArray> sampleHiddenGivenVisible(INDArray v) {


        INDArray h1Mean = propUp(v);
        INDArray h1Sample = Sampling.binomial(h1Mean, 1, rng);
        //apply dropout
        applyDropOutIfNecessary(h1Sample);
        return new Pair<>(h1Mean, h1Sample);

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
            INDArray wLearningRates = getAdaGrad().getLearningRates(gradient.getwGradient());
            INDArray z = reconstruct(input);

            //Scale the input and reconstruction to see the relative difference in absolute space
           /*
           Other current problems: the hbias mmul output diff is being calculated wrong.
           We should be able to calculate the w gradient with 1 mmul.
            */
            INDArray scaledInput = input.dup();
            Transforms.normalizeZeroMeanAndUnitVariance(scaledInput);
            Transforms.normalizeZeroMeanAndUnitVariance(z);
            INDArray outputDiff = z.sub(scaledInput);
            //changes in error relative to neurons
            INDArray delta = W.mmul(outputDiff);
            //hidden activations
            INDArray hBiasMean = z.sum(1).transpose();

            if(isUseAdaGrad()) {
                delta.muli(wLearningRates);
            }
            else
                delta.muli(lr);

            if(momentum != 0)
                delta.muli(momentum).add(delta.mul(1 - momentum));

            if(normalizeByInputRows)
                delta.divi(input.rows());


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
        INDArray chainStart = propUp(input);
        this.chainStart = chainStart;


		/*
		 * K steps of gibbs sampling. This is the positive phase of contrastive divergence.
		 *
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values
		 * or averages.
		 *
		 */

        INDArray nvSamples = null;
        INDArray hiddenMeans = chainStart;
        //contrastive divergence
        for(int i = 0; i < k; i++) {
            nvSamples = propDown(Sampling.binomial(eHid, 1, rng));
            hiddenMeans = propUp(nvSamples);
        }

		/*
		 * Update gradient parameters
		 */

        INDArray wGradient = NDArrays.create(W.shape());
        for(int i = 0; i < numFilters[0]; i++)
            for(int j = 0; j < numFilters[1]; j++) {
                wGradient.putSlice(j, Convolution.convn(input, NDArrays.reverse(eHid.slice(j)), Convolution.Type.VALID).subi(Convolution.convn(eVis, NDArrays.reverse(eHid.slice(j)), Convolution.Type.VALID)));
            }




        INDArray vBiasGradient = NDArrays.scalar((double) chainStart.sub(hiddenMeans).sum(1).sum(Integer.MAX_VALUE).element());

        //update rule: the expected values of the input - the negative samples adjusted by the learning rate
        INDArray  hBiasGradient = NDArrays.scalar((double) (input.sub(nvSamples)).sum(1).sum(Integer.MAX_VALUE).element());
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
    protected void applySparsity(INDArray hBiasGradient, double learningRate) {
        // dcSparse = self.lRate*self.sparseGain*(squeeze(self.sparsity -mean(mean(self.eHid0))));
        //self.c = self.c + dcSparse;
        if(sparsity != 0) {
            INDArray negMean = NDArrays.scalar(sparseGain * (sparsity -(double) chainStart.mean(1).mean(Integer.MAX_VALUE).element()));
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

    public INDArray getVisI() {
        return visI;
    }

    public void setVisI(INDArray visI) {
        this.visI = visI;
    }

    public INDArray getHidI() {
        return hidI;
    }

    public void setHidI(INDArray hidI) {
        this.hidI = hidI;
    }

    public void setW(INDArray w) {
        W = w;
    }

    public boolean isConvolutionInitCalled() {

        return convolutionInitCalled;
    }

    public void setConvolutionInitCalled(boolean convolutionInitCalled) {
        this.convolutionInitCalled = convolutionInitCalled;
    }

    public INDArray getChainStart() {
        return chainStart;
    }

    public void setChainStart(INDArray chainStart) {
        this.chainStart = chainStart;
    }

    public INDArray geteVis() {
        return eVis;
    }

    public void seteVis(INDArray eVis) {
        this.eVis = eVis;
    }

    public INDArray geteHid() {
        return eHid;
    }

    public void seteHid(INDArray eHid) {
        this.eHid = eHid;
    }

    public INDArray getwGradient() {
        return wGradient;
    }

    public void setwGradient(INDArray wGradient) {
        this.wGradient = wGradient;
    }

    public INDArray getvBiasGradient() {
        return vBiasGradient;
    }

    public void setvBiasGradient(INDArray vBiasGradient) {
        this.vBiasGradient = vBiasGradient;
    }

    public INDArray gethBiasGradient() {
        return hBiasGradient;
    }

    public void sethBiasGradient(INDArray hBiasGradient) {
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


    public INDArray getFeatureMap() {
        return featureMap;
    }

    public void setFeatureMap(INDArray featureMap) {
        this.featureMap = featureMap;
    }


    public INDArray getdWeights() {
        return dWeights;
    }

    public void setdWeights(INDArray dWeights) {
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
        public Builder withInput(INDArray input) {
            super.withInput(input);
            return this;
        }

        @Override
        public Builder asType(Class<RBM> clazz) {
            super.asType(clazz);
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
