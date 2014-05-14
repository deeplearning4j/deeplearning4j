package org.deeplearning4j.dbn;

import static org.deeplearning4j.util.MatrixUtil.downSample;
import static org.deeplearning4j.util.MatrixUtil.createBasedOn;
import static org.deeplearning4j.util.MatrixUtil.prod;
import static org.deeplearning4j.util.MatrixUtil.toMatrix;

import static org.deeplearning4j.util.Convolution.*;

import static org.deeplearning4j.util.MatrixUtil.rot;
import static org.jblas.ranges.RangeUtils.interval;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.*;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.learning.FourDTensorAdaGrad;
import org.deeplearning4j.rbm.ConvolutionalRBM;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.deeplearning4j.util.Convolution;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.RangeUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Convolutional Deep Belief Network
 * @author Adam Gibson
 */
public class ConvolutionalDBN extends BaseMultiLayerNetwork {


    //filter size per layer
    protected int[][] filterSizes;
    //the per datum input size (2d image dimensions)
    protected int[] inputSize;

    //rbm wise nfm
    protected int[] nFm;
    //layer wise stride (arr[k] is an array)
    protected int[][] stride;
    //layer wise filter number of filters
    protected int[][] numFilters;
    //sparse gain for each rbm
    protected double sparseGain;
    //layer type: convolution or subsampling
    protected LayerType[] layerTypes;
    public  static enum LayerType {
        SUBSAMPLE,CONVOLUTION
    }






    protected ConvolutionalDBN() {
    }

    protected ConvolutionalDBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers, RandomGenerator rng) {
        super(nIns, hiddenLayerSizes, nOuts, nLayers, rng);
    }

    protected ConvolutionalDBN(int nIn, int[] hiddenLayerSizes, int nOuts, int nLayers, RandomGenerator rng, DoubleMatrix input, DoubleMatrix labels) {
        super(nIn, hiddenLayerSizes, nOuts, nLayers, rng, input, labels);
    }

    /**
     * Train the network running some unsupervised
     * pretraining followed by SGD/finetune
     *
     * @param input       the input to train on
     * @param labels      the labels for the training examples(a matrix of the following format:
     *                    [0,1,0] where 0 represents the labels its not and 1 represents labels for the positive outcomes
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void trainNetwork(DoubleMatrix input, DoubleMatrix labels, Object[] otherParams) {

    }

    /**
     * Pretrain the network with the given parameters
     *
     * @param input       the input to train ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void pretrain(DoubleMatrix input, Object[] otherParams) {

    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @param input the input data to feed forward
     * @return the list of activations for each layer
     */
    @Override
    public List<DoubleMatrix> feedForward(DoubleMatrix input) {
        /* Number of tensors is equivalent to the number of mini batches */
        FourDTensor tensor = new FourDTensor(input,inputSize[0],inputSize[1],  input.rows / inputSize[0],input.rows / input.length);
        FourDTensor curr = tensor;
        List<DoubleMatrix> activations = new ArrayList<>();
        for(int i = 0; i < getnLayers(); i++) {
            ConvolutionalRBM r = (ConvolutionalRBM) getLayers()[i];
            DownSamplingLayer d = (DownSamplingLayer) getSigmoidLayers()[i];
            for(int j = 0; j < r.getNumFilters()[0]; j++) {

                int nInY = curr.rows();
                int nInX = curr.columns();
                int nInFm = curr.slices();
                int nObs = curr.getNumTensor();
                //equivalent to a 3d tensor: only one tensor
                FourDTensor featureMap = FourDTensor.zeros(r.getFmSize()[0],r.getFmSize()[1],1,nObs);
                for(int k = 0; j < r.getNumFilters()[0]; j++) {
                    featureMap.addi(conv2d(featureMap.getTensor(i),r.getW().getSliceOfTensor(j,i), Convolution.Type.VALID));
                }
                featureMap.addi(r.gethBias().get(i));
                r.getFeatureMap().setSlice(j,d.activate(featureMap));

                //put the down sampled
                d.getFeatureMap().setTensor(j,downSample(r.getFeatureMap().getTensor(j), MatrixUtil.toMatrix(r.getStride())));


            }

            activations.add(d.getFeatureMap());

        }

        activations.add(output(activations.get(activations.size() - 1)));


        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    @Override
    public List<DoubleMatrix> feedForward() {
        return feedForward(input);
    }

    @Override
    protected void computeDeltas(List<Pair<DoubleMatrix, DoubleMatrix>> deltaRet) {
        ActivationFunction a = getSigmoidLayers()[0].getActivationFunction();
        ActivationFunction softMaxDerivative = Activations.softmax();
        List<DoubleMatrix> activations = feedForward();

        /**
         * Prediction will actually be a tensor, need to map this out
         */
        DoubleMatrix error = labels.sub(activations.get(activations.size() - 1)).neg().mul(softMaxDerivative.applyDerivative(activations.get(activations.size() - 1)));
        //should this be a 4d tensor?
        DoubleMatrix es = outputLayer.getW().transpose().mmul(error);
        DownSamplingLayer d = (DownSamplingLayer) getSigmoidLayers()[getSigmoidLayers().length - 1];
        DoubleMatrix shape = d.getFeatureMap().shape();
        ConvolutionalRBM rbm = (ConvolutionalRBM) getLayers()[getnLayers() - 1];
        DoubleMatrix[] errorSignals = new DoubleMatrix[getnLayers()];
        DoubleMatrix[] biasGradients = new DoubleMatrix[getnLayers()];

        FourDTensor layerErrorSignal = FourDTensor.zeros((int) shape.get(0),(int) shape.get(1),(int) shape.get(2),(int) shape.get(3));
        errorSignals[errorSignals.length - 1] = es;
        //initial hidden layer error signal

        int nMap = (int) shape.get(0) * (int) shape.get(1);

        //translate in to a 2d feature map
        for(int i = 0; i < rbm.getNumFilters()[0]; i++) {
           /*
             These will be different slices of the tensor
            */
            DoubleMatrix subSlice = es.get(RangeUtils.interval(i * nMap,(i + 1)* nMap),RangeUtils.interval(0,es.columns));
            Tensor reshaped = MatrixUtil.reshape(subSlice,(int)shape.get(0),(int)shape.get(1),(int) shape.get(3));
            layerErrorSignal.setTensor(i,reshaped);

        }

        errorSignals[errorSignals.length - 2] = layerErrorSignal;

        for(int i = getnLayers() -2; i >= 0; i--) {
            DownSamplingLayer layer = (DownSamplingLayer) getSigmoidLayers()[i];
            DoubleMatrix shape2 = d.getFeatureMap().shape();
            ConvolutionalRBM r2 = (ConvolutionalRBM) getLayers()[i];
            DownSamplingLayer forwardDownSamplingLayer = (DownSamplingLayer) getSigmoidLayers()[i + 1];
            ConvolutionalRBM forwardRBM = (ConvolutionalRBM) getLayers()[i + 1];
            int[] stride = forwardRBM.getStride();

            FourDTensor propErrorSignal = FourDTensor.zeros((int) shape2.get(0),(int) shape2.get(1),(int) shape2.get(2),(int) shape2.get(3));

            //handle subsampling layer first
            for(int k = 0; k < layer.getNumFeatureMaps(); k++) {
                DoubleMatrix rotFilter = rot(forwardRBM.getW().getSliceOfTensor(i,k));
                FourDTensor tensor = (FourDTensor) errorSignals[i + 1];
                Tensor currEs = tensor.getTensor(k);
                propErrorSignal.addi(conv2d(currEs,rotFilter,Type.FULL));

            }


            errorSignals[i] = propErrorSignal;

            DoubleMatrix mapSize = forwardRBM.getFeatureMap().shape();
            FourDTensor rbmEs = FourDTensor.zeros((int) mapSize.get(0),(int) mapSize.get(1),(int) mapSize.get(2),(int) mapSize.get(3));
            for(int k = 0; k < rbm.getNumFilters()[0]; k++) {
                Tensor propEs = MatrixUtil.upSample(forwardDownSamplingLayer.getFeatureMap().getTensor(k),new Tensor(toMatrix(new int[]{stride[0],stride[1],1,1}).div(prod(toMatrix(stride)))));
                rbmEs.setTensor(k,propEs);
            }

            errorSignals[i - 1] = rbmEs;



        }



        //now calculate the gradients

        for(int i = getnLayers() -2; i >= 0; i--) {
            ConvolutionalRBM r2 = (ConvolutionalRBM) getLayers()[i];
            ConvolutionalRBM prevRBM = (ConvolutionalRBM) getLayers()[i - 1];

            FourDTensor errorSignal = (FourDTensor) errorSignals[i - 1];
            DoubleMatrix biasGradient = new DoubleMatrix(1,r2.getNumFilters()[0]);
            for(int j = 0; j < r2.getNumFilters()[0]; j++) {
                Tensor es2 = errorSignal.getTensor(j);
                for(int k = 0; k < prevRBM.getNumFilters()[0]; k++)  {

                    //figure out what to do wrt error signal for each neural net here.
                    Tensor flipped = MatrixUtil.flipDimMultiDim(prevRBM.getFeatureMap().getTensor(k));

                    DoubleMatrix dedFilter = conv2d(flipped,es2,Type.VALID);
                    r2.getdWeights().put(j,k,dedFilter);
                }

                biasGradient.put(j,es.columnSums().div(errorSignal.numTensors()).sum());

            }
            biasGradients[i] = biasGradient;

        }


        for(int i = 0; i < biasGradients.length; i++) {
            deltaRet.add(new Pair<>(errorSignals[i],biasGradients[i]));
        }

        //output layer gradients
        deltaRet.add(new Pair<>(errorSignals[errorSignals.length - 1].mmul(outputLayer.getInput()),errorSignals[errorSignals.length - 1].columnMeans()));

    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     *
     * @param revert the best network so far
     * @param lr     the learning rate to use for training
     * @param epoch  the epoch to use
     * @return whether the training should converge or not
     */
    @Override
    protected void backPropStep(BaseMultiLayerNetwork revert, double lr, int epoch) {
        //feedforward to compute activations
        //initial error


        //precompute deltas
        List<Pair<DoubleMatrix,DoubleMatrix>> deltas = new ArrayList<>();
        //compute derivatives and gradients given activations
        computeDeltas(deltas);


        for(int l = 1; l < getnLayers(); l++) {
            ConvolutionalRBM r = (ConvolutionalRBM) getLayers()[l];
            ConvolutionalRBM prevR = (ConvolutionalRBM) getLayers()[l - 1];

            FourDTensor wGradient =  (FourDTensor) deltas.get(l).getFirst();
            DoubleMatrix biasGradient =  deltas.get(l).getSecond();
            FourDTensorAdaGrad wAdaGrad = (FourDTensorAdaGrad) r.getAdaGrad();
            DoubleMatrix biasLearningRates = null;
            if(useAdaGrad)
                biasLearningRates = layers[l].gethBiasAdaGrad().getLearningRates(biasGradient);


            for(int m = 0; m < r.getNumFilters()[0]; m++) {
                if(useAdaGrad)
                    biasGradient.put(m,biasLearningRates.get(m) *  r.gethBias().get(m));

                else
                    biasGradient.put(m,lr *  r.gethBias().get(m));
                for(int n = 0; n < prevR.getNumFilters()[0]; n++) {
                    if(useRegularization)  {
                        DoubleMatrix penalty = r.getFeatureMap().getSliceOfTensor(m,n).mul(l2) ;
                        if(useAdaGrad) {
                            DoubleMatrix learningRates = wAdaGrad.getLearningRates(m,n,wGradient.getSliceOfTensor(m,n));
                            penalty.muli(learningRates);

                        }
                        else
                            penalty.muli(lr);
                        wGradient.put(m,n,wGradient.getSliceOfTensor(m,n).mul(penalty));

                    }
                    //hmm: https://github.com/agibsonccc/medal/blob/master/models/mlcnn.m#L422
                    // wGradient.subi()

                    //r.getW().put(m,n,r.getW().getSliceOfTensor(m,n).sub(wGradient.getSliceOfTensor(m,n)));

                }


            }


            DoubleMatrix gradientChange = deltas.get(l).getFirst();
            //get the gradient
            if(isUseAdaGrad())
                gradientChange.muli(getLayers()[l].getAdaGrad().getLearningRates(gradientChange));

            else
                gradientChange.muli(lr);

            //l2
            if(useRegularization)
                gradientChange.muli(getLayers()[l].getW().mul(l2));

            if(momentum != 0)
                gradientChange.muli(momentum);

            if(isNormalizeByInputRows())
                gradientChange.divi(input.rows);

            //update W
            getLayers()[l].getW().subi(gradientChange);
            getSigmoidLayers()[l].setW(layers[l].getW());


            //update hidden bias
            DoubleMatrix deltaColumnSums = deltas.get(l + 1).getSecond().columnSums();

            if(sparsity != 0)
                deltaColumnSums = MatrixUtil.scalarMinus(sparsity, deltaColumnSums);

            if(useAdaGrad)
                deltaColumnSums.muli(layers[l].gethBiasAdaGrad().getLearningRates(deltaColumnSums));
            else
                deltaColumnSums.muli(lr);

            if(momentum != 0)
                deltaColumnSums.muli(momentum);

            if(isNormalizeByInputRows())
                deltaColumnSums.divi(input.rows);


            getLayers()[l].gethBias().subi(deltaColumnSums);
            getSigmoidLayers()[l].setB(getLayers()[l].gethBias());
        }

        DoubleMatrix logLayerGradient = deltas.get(getnLayers()).getFirst();
        DoubleMatrix biasGradient = deltas.get(getnLayers()).getSecond().columnSums();

        if(momentum != 0)
            logLayerGradient.muli(momentum);


        if(useAdaGrad)
            logLayerGradient.muli(outputLayer.getAdaGrad().getLearningRates(logLayerGradient));


        else
            logLayerGradient.muli(lr);

        if(isNormalizeByInputRows())
            logLayerGradient.divi(input.rows);



        if(momentum != 0)
            biasGradient.muli(momentum);

        if(useAdaGrad)
            biasGradient.muli(outputLayer.getBiasAdaGrad().getLearningRates(biasGradient));
        else
            biasGradient.muli(lr);

        if(isNormalizeByInputRows())
            biasGradient.divi(input.rows);


        getOutputLayer().getW().subi(logLayerGradient);

        if(getOutputLayer().getB().length == biasGradient.length)
            getOutputLayer().getB().subi(biasGradient);
        else
            getOutputLayer().getB().subi(biasGradient.mean());
    }


    /**
     * Label the probabilities of the input
     *
     * @param x the input to label
     * @return a vector of probabilities
     * given each label.
     * <p/>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    @Override
    public DoubleMatrix output(DoubleMatrix x) {
        DownSamplingLayer d = (DownSamplingLayer)  getSigmoidLayers()[getSigmoidLayers().length - 1];
        FourDTensor lastLayer = d.getFeatureMap();
        int nY = lastLayer.getRows();
        int nX = lastLayer.getColumns();
        int nM = lastLayer.slices();
        int noBs = lastLayer.getNumTensor();

        int nMap = nY * nX;

        DoubleMatrix features = new DoubleMatrix(nMap * nM,noBs);


        for(int j = 0; j < d.getNumFeatureMaps(); j++) {
            Tensor map = d.getFeatureMap().getTensor(j);
            features.put(interval(j * nMap + 1,j * nMap),interval(0,features.columns),map.reshape(nMap,noBs));
        }




        return createBasedOn(outputLayer.output(features),features);
    }

    /**
     * Backpropagation of errors for weights
     *
     * @param lr     the learning rate to use
     * @param epochs the number of epochs to iterate (this is already called in finetune)
     */
    @Override
    public void backProp(double lr, int epochs) {
        super.backProp(lr, epochs);
    }

    @Override
    public void init() {
        Tensor input = (Tensor) this.input;
        if(!(rng instanceof SynchronizedRandomGenerator))
            rng = new SynchronizedRandomGenerator(rng);
        if(getnLayers() < 1)
            throw new IllegalStateException("Unable to create network layers; number specified is less than 1");

        if(this.dist == null)
            dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

        this.layers = new NeuralNetwork[getnLayers()];

        // construct multi-layer
        int nInY,nInX,nInFM,nFm = -1;
        for(int i = 0; i < getnLayers(); i++) {
            ConvolutionalRBM prevLayer = (ConvolutionalRBM) getLayers()[i];

            if(i == 0) {
                nInY = input.rows();
                nInX = input.columns();
                nInFM = input.slices();
            }

            else {
                nInX = prevLayer.getFmSize()[1];
                nInY = prevLayer.getFmSize()[0];
                nInFM = prevLayer.getFmSize()[0];
            }

            nFm = this.nFm[i];
            DoubleMatrix filterSize = MatrixUtil.toMatrix(this.filterSizes[i]);
            DoubleMatrix fmSize = MatrixUtil.toMatrix(new int[]{nInY,nInX}).sub(filterSize).add(1);
            double prodFilterSize = MatrixUtil.prod(filterSize);
            DoubleMatrix stride = MatrixUtil.toMatrix(this.stride[i]);
            double fanIn = nInFM * prodFilterSize;
            double fanOut = nFm * prodFilterSize;

            double range = 2 * FastMath.sqrt(6 / fanIn + fanOut);
            FourDTensor W = FourDTensor.rand((int) filterSize.get(0),(int) filterSize.get(1),nInFM,nFm,dist).mul(range);

            ConvolutionalRBM r = new ConvolutionalRBM.Builder().withDistribution(getDist())
                    .withDropOut(getDropOut()).withOptmizationAlgo(getOptimizationAlgorithm())
                    .withFilterSize(MatrixUtil.toInts(filterSize)).withInput(input).numberOfVisible(1)
                    .useAdaGrad(isUseAdaGrad()).normalizeByInputRows(normalizeByInputRows)
                    .numHidden(1).withHBias(DoubleMatrix.zeros(nFm,1)).withMomentum(getMomentum())
                    .withL2(getL2()).useRegularization(isUseRegularization())
                    .withRandom(rng).withLossFunction(getLossFunction()).withFmSize(MatrixUtil.toInts(fmSize))
                    .withStride(this.stride[i]).withNumFilters(new int[]{nFm,nFm})
                    .withSparseGain(sparseGain).withSparsity(getSparsity())
                    .withWeights(W).build();
            this.layers[i] = r;


            DownSamplingLayer d = new DownSamplingLayer.Builder()
                    .dist(dist).withStride(this.stride[i])
                    .withFmSize(MatrixUtil.floor(MatrixUtil.toMatrix(r.getFmSize()).div(stride)))
                    .numFeatureMaps(nFm).withBias(r.gethBias())
                    .withRng(rng).build();

            this.sigmoidLayers[i] = d;


        }

        ConvolutionalRBM r= (ConvolutionalRBM) getLayers()[getLayers().length - 1];

        int nFmIn =  r.getNumFilters()[0];
        int nOuts = r.getFmSize()[0] * r.getFmSize()[1] * nFmIn;


        // layer for output using OutputLayer
        this.outputLayer = new OutputLayer.Builder()
                .useAdaGrad(useAdaGrad).optimizeBy(getOptimizationAlgorithm())
                .normalizeByInputRows(normalizeByInputRows)
                .useRegularization(useRegularization)
                .numberOfInputs(nFmIn)
                .numberOfOutputs(nOuts).withL2(l2).build();

        synchonrizeRng();

        applyTransforms();
        initCalled = true;

    }

    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overridden
     * for other kinds of hidden units
     *
     * @param index
     * @param nIn        the number of inputs
     * @param nOut       the number of outputs
     * @param activation the activation function for the layer
     * @param rng        the rng to use for sampling
     * @param layerInput the layer starting input
     * @param dist       the probability distribution to use
     *                   for generating weights
     * @return a hidden layer with the given parameters
     */
    @Override
    public DownSamplingLayer createHiddenLayer(int index, int nIn, int nOut, ActivationFunction activation, RandomGenerator rng, DoubleMatrix layerInput, RealDistribution dist) {
        ConvolutionalRBM r = (ConvolutionalRBM) getLayers()[index - 1];
        DownSamplingLayer layer = new DownSamplingLayer.Builder().dist(dist).withInput(layerInput)
                .withFmSize(MatrixFunctions.floor(MatrixUtil.toMatrix(r.getFmSize())).div(MatrixUtil.toMatrix(stride[index])))
                .numFeatureMaps(nFm[index])
                .nIn(nIn).nOut(nOut).withActivation(activation).build();
        return layer;
    }

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.rbm.RBM} for continuous inputs.
     * <p/>
     * Please be sure to call super.initializeNetwork
     * <p/>
     * to handle the passing of baseline parameters such as fanin
     * and rendering.
     *
     * @param input    the input to the layer
     * @param nVisible the number of visible inputs
     * @param nHidden  the number of hidden units
     * @param W        the weight vector
     * @param hbias    the hidden bias
     * @param vBias    the visible bias
     * @param rng      the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
     * @param index    the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.rbm.RBM}
     */
    @Override
    public NeuralNetwork createLayer(DoubleMatrix input, int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vBias, RandomGenerator rng, int index) {
        ConvolutionalRBM r = new ConvolutionalRBM.Builder().withDistribution(getDist())
                .withDropOut(getDropOut()).withOptmizationAlgo(getOptimizationAlgorithm())
                .withFilterSize(filterSizes[index]).withInput(input).numberOfVisible(nVisible)
                .useAdaGrad(isUseAdaGrad()).normalizeByInputRows(normalizeByInputRows)
                .numHidden(nHidden).withHBias(hbias).withMomentum(getMomentum())
                .withL2(getL2()).useRegularization(isUseRegularization())
                .withRandom(rng).withLossFunction(getLossFunction())
                .withStride(stride[index]).withNumFilters(numFilters[index])
                .withSparseGain(sparseGain).withSparsity(getSparsity())
                .withWeights(W).build();

        return r;
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[numLayers];
    }
}
