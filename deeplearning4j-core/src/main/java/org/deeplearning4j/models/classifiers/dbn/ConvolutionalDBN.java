package org.deeplearning4j.models.classifiers.dbn;



import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.nn.*;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.DownSamplingLayer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.nd4j.linalg.learning.AdaGrad;
import org.deeplearning4j.models.featuredetectors.rbm.ConvolutionalRBM;


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
    protected float sparseGain;
    //layer type: convolution or subsampling
    protected LayerType[] layerTypes;

    /**
     * Transform the data based on the model's output.
     * This can be anything from a number to reconstructions.
     *
     * @param data the data to transform
     * @return the transformed data
     */
    @Override
    public INDArray transform(INDArray data) {
        return null;
    }

    /**
     * Fit the model to the given data
     *
     * @param data   the data to fit the model to
     * @param params the params (mixed values)
     */
    @Override
    public void fit(INDArray data, Object[] params) {

    }

    public  static enum LayerType {
        SUBSAMPLE,CONVOLUTION
    }






    protected ConvolutionalDBN() {
    }



    /**
     * Pretrain the network with the given parameters
     *
     * @param input       the input to iterate ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void pretrain(INDArray input, Object[] otherParams) {

    }

    /**
     * Pretrain with a data applyTransformToDestination iterator.
     * This will run through each neural net at a time and iterate on the input.
     *
     * @param iter        the iterator to use
     * @param otherParams
     */
    @Override
    public void pretrain(DataSetIterator iter, Object[] otherParams) {

    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @param input the input data to feed forward
     * @return the list of activations for each layer
     */
    @Override
    public List<INDArray> feedForward(INDArray input) {
        /* Number of tensors is equivalent to the number of mini batches */
        INDArray tensor = Nd4j.create(new int[]{inputSize[0],inputSize[1],  input.rows() / inputSize[0],input.rows() / input.length()});
        INDArray curr = tensor;
        List<INDArray> activations = new ArrayList<>();
        for(int i = 0; i < getnLayers(); i++) {
            ConvolutionalRBM r = (ConvolutionalRBM) getNeuralNets()[i];
            DownSamplingLayer d = (DownSamplingLayer) getNeuralNets()[i];
            for(int j = 0; j < r.getNumFilters()[0]; j++) {

                int nObs = curr.slices();
                //equivalent to a 3d tensor: only one tensor
                INDArray featureMap = Nd4j.zeros(new int[]{r.getFmSize()[0],r.getFmSize()[1],1,nObs});
                for(int k = 0; j < r.getNumFilters()[0]; j++) {
                    featureMap.addi(Convolution.convn(featureMap.slice(i), r.getW().slice(j, i), Convolution.Type.VALID));
                }
                featureMap.addi(r.gethBias().getScalar(i));
                r.getFeatureMap().putSlice(j, d.activate(featureMap));

                //put the down sampled
                d.getFeatureMap().putSlice(j, Transforms.downSample(r.getFeatureMap().slice(j),r.getStride()));


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
    public List<INDArray> feedForward() {
        return feedForward(input);
    }

    @Override
    protected void computeDeltas(List<INDArray> deltaRet) {
        ActivationFunction softMaxDerivative = Activations.softMaxRows();
        List<INDArray> activations = feedForward();

        /**
         * Prediction will actually be a tensor, need to map this out
         */
        INDArray error = labels.sub(activations.get(activations.size() - 1)).neg().mul(softMaxDerivative.applyDerivative(activations.get(activations.size() - 1)));
        //should this be a 4d tensor?
        INDArray es = getOutputLayer().getW().transpose().mmul(error);
        DownSamplingLayer d = (DownSamplingLayer) getNeuralNets()[getNeuralNets().length - 1];
        ConvolutionalRBM rbm = (ConvolutionalRBM) getNeuralNets()[getnLayers() - 1];
        INDArray[] errorSignals = new INDArray[getnLayers()];
        INDArray[] biasGradients = new INDArray[getnLayers()];

        INDArray layerErrorSignal = Nd4j.zeros(d.getFeatureMap().shape());
        errorSignals[errorSignals.length - 1] = es;
        //initial hidden layer error signal

        int nMap = layerErrorSignal.size(0)*  layerErrorSignal.size(1);

        //translate in to a 2d feature map
        for(int i = 0; i < rbm.getNumFilters()[0]; i++) {
           /*
             These will be different slices of the tensor
            */
            INDArray subSlice = es.get(NDArrayIndex.interval(i * nMap, (i + 1) * nMap),NDArrayIndex.interval(0,es.columns()));
            INDArray reshaped = subSlice.reshape(d.getFeatureMap().shape());
            layerErrorSignal.putSlice(i, reshaped);

        }

        errorSignals[errorSignals.length - 2] = layerErrorSignal;

        for(int i = getnLayers() -2; i >= 0; i--) {
            DownSamplingLayer layer = (DownSamplingLayer) getNeuralNets()[i];
            ConvolutionalRBM r2 = (ConvolutionalRBM) getNeuralNets()[i];
            DownSamplingLayer forwardDownSamplingLayer = (DownSamplingLayer) getNeuralNets()[i + 1];
            ConvolutionalRBM forwardRBM = (ConvolutionalRBM) getNeuralNets()[i + 1];
            int[] stride = forwardRBM.getStride();

            INDArray propErrorSignal = Nd4j.zeros(d.getFeatureMap().shape());

            //handle subsampling layer first
            for(int k = 0; k < layer.getNumFeatureMaps(); k++) {
                INDArray rotFilter = Nd4j.rot(forwardRBM.getW().slice(i).slice(k));
                INDArray tensor =   errorSignals[i + 1];
                INDArray currEs = tensor.slice(k);
                propErrorSignal.addi(Convolution.convn(currEs, rotFilter, Convolution.Type.FULL));

            }


            errorSignals[i] = propErrorSignal;

            INDArray mapSize = forwardRBM.getFeatureMap();
            INDArray rbmEs = Nd4j.zeros(mapSize.shape());
            for(int k = 0; k < rbm.getNumFilters()[0]; k++) {
                INDArray propEs = Transforms.upSample(forwardDownSamplingLayer.getFeatureMap().slice(k),
                        ArrayUtil.toNDArray(new int[]{stride[0],stride[1],1,1})).divi(ArrayUtil.prod(stride));
                rbmEs.putSlice(k, propEs);
            }

            errorSignals[i - 1] = rbmEs;



        }



        //now calculate the gradients

        for(int i = getnLayers() -2; i >= 0; i--) {
            ConvolutionalRBM r2 = (ConvolutionalRBM) getNeuralNets()[i];
            ConvolutionalRBM prevRBM = (ConvolutionalRBM) getNeuralNets()[i - 1];

            INDArray errorSignal = errorSignals[i - 1];
            INDArray biasGradient = Nd4j.create(1,r2.getNumFilters()[0]);
            for(int j = 0; j < r2.getNumFilters()[0]; j++) {
                INDArray es2 = errorSignal.slice(j);
                for(int k = 0; k < prevRBM.getNumFilters()[0]; k++)  {

                    //figure out what to do wrt error signal for each neural net here.
                    INDArray flipped = Nd4j.reverse(prevRBM.getFeatureMap().slice(k));

                    INDArray dedFilter = Convolution.convn(flipped, es2, Convolution.Type.VALID);
                    r2.getdWeights().put(j,k,dedFilter);
                }

                biasGradient.put(j, es.sum(1).div(errorSignal.slices()).sum(Integer.MAX_VALUE));

            }
            biasGradients[i] = biasGradient;

        }


        for(int i = 0; i < biasGradients.length; i++) {
            deltaRet.add(errorSignals[i]);
        }

        //output layer gradients
        deltaRet.add(errorSignals[errorSignals.length - 1].mmul(getOutputLayer().getInput()));

    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last neuralNets weights
     * to revert to in case of convergence, the learning rate being used to iterate
     * and the current epoch
     *
     * @param revert the best network so far
     * @param epoch  the epoch to use
     * @return whether the training should converge or not
     */
    //@Override
    protected void backPropStep(BaseMultiLayerNetwork revert, int epoch) {
        //feedforward to compute activations
        //initial error


        //precompute deltas
        List<INDArray> deltas = new ArrayList<>();
        //compute derivatives and gradients given activations
        computeDeltas(deltas);


        for(int l = 1; l < getnLayers(); l++) {
            ConvolutionalRBM r = (ConvolutionalRBM) getNeuralNets()[l];
            ConvolutionalRBM prevR = (ConvolutionalRBM) getNeuralNets()[l - 1];
            NeuralNetConfiguration conf = r.conf();
            INDArray wGradient =   deltas.get(l);
            INDArray biasGradient =  deltas.get(l).mean(1);

            INDArray biasLearningRates = null;
            AdaGrad wAdaGrad= r.getAdaGrad();

            if(conf.isUseAdaGrad())
                biasLearningRates = neuralNets[l].gethBiasAdaGrad().getLearningRates(biasGradient);


            for(int m = 0; m < r.getNumFilters()[0]; m++) {
                if(conf.isUseAdaGrad())
                    biasGradient.put(m,biasLearningRates.getScalar(m).muli(r.gethBias().getScalar(m)));

                else
                    biasGradient.put(m,r.gethBias().getScalar(m).muli(conf.getLr()));
                for(int n = 0; n < prevR.getNumFilters()[0]; n++) {
                    if(conf.isUseRegularization())  {
                        INDArray penalty = r.getFeatureMap().slice(m, n).mul(conf.getL2()) ;
                        if(conf.isUseAdaGrad()) {
                            INDArray learningRates = wAdaGrad.getLearningRates(wGradient.slice(m).slice(n));
                            penalty.muli(learningRates);

                        }
                        else
                            penalty.muli(conf.getLr());
                        wGradient.put(m, n, wGradient.slice(m).slice(n).mul(penalty));

                    }

                }


            }


            INDArray gradientChange = deltas.get(l);
            //getFromOrigin the gradient
            if(conf.isUseAdaGrad())
                gradientChange.muli(getNeuralNets()[l].getAdaGrad().getLearningRates(gradientChange));

            else
                gradientChange.muli(conf.getLr());

            //l2
            if(conf.isUseRegularization())
                gradientChange.muli(getNeuralNets()[l].getW().mul(conf.getL2()));

            if(conf.getMomentum() != 0)
                gradientChange.muli(conf.getMomentum());

            gradientChange.divi(input.rows());

            //update W
            getNeuralNets()[l].getW().subi(gradientChange);
            getNeuralNets()[l].setW(neuralNets[l].getW());


            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).mean(1);

            if(conf.getSparsity() != 0)
                deltaColumnSums = deltaColumnSums.rsubi(conf.getSparsity());

            if(conf.isUseAdaGrad())
                deltaColumnSums.muli(neuralNets[l].gethBiasAdaGrad().getLearningRates(deltaColumnSums));
            else
                deltaColumnSums.muli(conf.getLr());

            if(conf.getMomentum() != 0)
                deltaColumnSums.muli(conf.getMomentum());

            deltaColumnSums.divi(input.rows());


            getNeuralNets()[l].gethBias().subi(deltaColumnSums);
            getLayers()[l].setB(getNeuralNets()[l].gethBias());
        }

        INDArray logLayerGradient = deltas.get(getnLayers());
        INDArray biasGradient = deltas.get(getnLayers()).mean(1);
        NeuralNetConfiguration conf = getOutputLayer().conf();

        if(conf.getMomentum() != 0)
            logLayerGradient.muli(conf.getMomentum());


        if(conf.isUseAdaGrad())
            logLayerGradient.muli(getOutputLayer().getAdaGrad().getLearningRates(logLayerGradient));


        else
            logLayerGradient.muli(conf.getLr());

        logLayerGradient.divi(input.rows());



        if(conf.getMomentum() != 0)
            biasGradient.muli(conf.getMomentum());

        if(conf.isUseAdaGrad())
            biasGradient.muli(getOutputLayer().getBiasAdaGrad().getLearningRates(biasGradient));
        else
            biasGradient.muli(conf.getLr());

        biasGradient.divi(input.rows());


        getOutputLayer().getW().subi(logLayerGradient);

        if(getOutputLayer().getB().length() == biasGradient.length())
            getOutputLayer().getB().subi(biasGradient);
        else
            getOutputLayer().getB().subi(biasGradient.mean(Integer.MAX_VALUE));
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
    public INDArray output(INDArray x) {
        DownSamplingLayer d = (DownSamplingLayer)  getNeuralNets()[getNeuralNets().length - 1];
        INDArray lastLayer = d.getFeatureMap();
        int nY = lastLayer.shape()[3];
        int nX = lastLayer.shape()[2];
        int nM = lastLayer.shape()[1];
        int noBs = lastLayer.slices();

        int nMap = nY * nX;

        INDArray features = Nd4j.create(nMap * nM, noBs);


        for(int j = 0; j < d.getNumFeatureMaps(); j++) {
            INDArray map = d.getFeatureMap().slice(j);
            features.put(new NDArrayIndex[] { NDArrayIndex.interval(j * nMap + 1, j * nMap),NDArrayIndex.interval(0, features.columns())},map.reshape(nMap,noBs));
        }




        return getOutputLayer().output(features);
    }



    /**
     * Backpropagation of errors for weights
     *
     * @param lr     the learning rate to use
     * @param epochs the number of epochs to iterate (this is already called in finetune)
     */
    @Override
    public void backProp(float lr, int epochs) {
        super.backProp(lr, epochs);
    }

    @Override
    public void init() {

        this.neuralNets = new NeuralNetwork[getnLayers()];

        // construct multi-layer
        int nInY,nInX,nInFM,nFm = -1;
        for(int i = 0; i < getnLayers(); i++) {
            ConvolutionalRBM prevLayer = (ConvolutionalRBM) getNeuralNets()[i];

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
            INDArray filterSize = ArrayUtil.toNDArray(this.filterSizes[i]);
            INDArray fmSize = ArrayUtil.toNDArray(new int[]{nInY,nInX}).sub(filterSize).add(1);
            float prodFilterSize = ArrayUtil.prod(this.filterSizes[i]);
            INDArray stride = ArrayUtil.toNDArray(this.stride[i]);
            float fanIn = nInFM * prodFilterSize;
            float fanOut = nFm * prodFilterSize;

            float range = 2 * (float) FastMath.sqrt(6 / fanIn + fanOut);
            INDArray W = Nd4j.rand(new int[]{(int) filterSize.getScalar(0).element(),(int) filterSize.getScalar(1).element(),nInFM,nFm},layerWiseConfigurations.get(i).getDist()).mul(range);

            ConvolutionalRBM r = new ConvolutionalRBM.Builder()
                    .withFilterSize(ArrayUtil.toInts(filterSize)).withInput(input)
                    .withHBias(Nd4j.zeros(nFm, 1))
                    .withFmSize(ArrayUtil.toInts(fmSize))
                    .withStride(this.stride[i]).withNumFilters(new int[]{nFm, nFm})
                    .withSparseGain(sparseGain)
                    .withWeights(W).build();
            this.neuralNets[i] = r;


            DownSamplingLayer d = new DownSamplingLayer.Builder()
                    .withStride(this.stride[i])
                    .withFmSize(Transforms.floor(ArrayUtil.toNDArray(r.getFmSize()).div(stride)))
                    .numFeatureMaps(nFm).withBias(r.gethBias())
                    .build();

            this.layers[i] = d;


        }

        ConvolutionalRBM r= (ConvolutionalRBM) getNeuralNets()[getNeuralNets().length - 1];

        int nFmIn =  r.getNumFilters()[0];
        int nOuts = r.getFmSize()[0] * r.getFmSize()[1] * nFmIn;
        layerWiseConfigurations.get(layerWiseConfigurations.size() - 1).setnIn(nFmIn);
        layerWiseConfigurations.get(layerWiseConfigurations.size() - 1).setnOut(nOuts);

        // layer for output using OutputLayer
        this.layers[layers.length - 1] = new OutputLayer.Builder()
                .configure(layerWiseConfigurations.get(layerWiseConfigurations.size() - 1))
                .build();


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
     * @param layerInput the layer starting input
     * @return a hidden layer with the given parameters
     */
    @Override
    public DownSamplingLayer createHiddenLayer(int index,  INDArray layerInput) {
        ConvolutionalRBM r = (ConvolutionalRBM) getNeuralNets()[index - 1];
        DownSamplingLayer layer = new DownSamplingLayer.Builder() .withInput(layerInput)
                .withFmSize(Transforms.floor(ArrayUtil.toNDArray(r.getFmSize())).div(ArrayUtil.toNDArray(stride[index])))
                .numFeatureMaps(nFm[index])
                .build();
        return layer;
    }

    @Override
    public Layer createHiddenLayer(int index, int nIn, int nOut, INDArray layerInput) {
        return null;
    }

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.models.classifiers.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.models.featuredetectors.rbm.RBM} for continuous inputs.
     * <p/>
     * Please be sure to call super.initializeNetwork
     * <p/>
     * to handle the passing of baseline parameters such as fanin
     * and rendering.
     *
     * @param input    the input to the layer
     * @param W        the weight vector
     * @param hbias    the hidden bias
     * @param vBias    the visible bias
     * @param index    the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.models.featuredetectors.rbm.RBM}
     */
    @Override
    public NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hbias, INDArray vBias, int index) {
        ConvolutionalRBM r = new ConvolutionalRBM.Builder()
                .withFilterSize(filterSizes[index]).withInput(input)
                .withHBias(hbias).configure(layerWiseConfigurations.get(index))

                .withStride(stride[index]).withNumFilters(numFilters[index])
                .withSparseGain(sparseGain)
                .withWeights(W).build();

        return r;
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[numLayers];
    }
}
