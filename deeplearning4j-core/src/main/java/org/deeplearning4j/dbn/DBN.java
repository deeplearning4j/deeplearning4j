package org.deeplearning4j.dbn;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.*;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.transformation.MatrixTransform;
import org.deeplearning4j.util.Dl4jReflection;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;


/**
 * Deep Belief Network. This is a MultiLayer Perceptron Model
 * using Restricted Boltzmann Machines.
 *  See Hinton's practical guide to RBMs for great examples on
 *  how to train and tune parameters.
 *
 * @author Adam Gibson
 *
 */
public class DBN extends BaseMultiLayerNetwork {

    private static final long serialVersionUID = -9068772752220902983L;
    private static Logger log = LoggerFactory.getLogger(DBN.class);
    private RBM.VisibleUnit visibleUnit;
    private RBM.HiddenUnit hiddenUnit;
    private Map<Integer,RBM.VisibleUnit> visibleUnitByLayer = new HashMap<>();
    private Map<Integer,RBM.HiddenUnit> hiddenUnitByLayer = new HashMap<>();


    public DBN() {}


    public DBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers,
               RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
        super(nIns, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);
    }



    public DBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers,
               RandomGenerator rng) {
        super(nIns, hiddenLayerSizes, nOuts, nLayers, rng);
    }


    /**
     *
     * @param input input examples
     * @param labels output labels
     * @param otherParams
     *
     * (int)    k
     * (double) learningRate
     * (int) epochs
     *
     * Optional:
     * (double) finetune lr
     * (int) finetune epochs
     *
     */
    @Override
    public void trainNetwork(DoubleMatrix input, DoubleMatrix labels,
                             Object[] otherParams) {
        int k = (Integer) otherParams[0];
        double lr = (Double) otherParams[1];
        int epochs = (Integer) otherParams[2];
        pretrain(input,k,lr,epochs);
        if(otherParams.length < 3)
            finetune(labels, lr, epochs);
        else {
            double finetuneLr = otherParams.length > 3 ? (double) otherParams[3] : lr;
            int finetuneEpochs = otherParams.length > 4 ? (int) otherParams[4] : epochs;
            finetune(labels,finetuneLr,finetuneEpochs);
        }
    }



    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overriden
     * for other kinds of hidden units
     * @param nIn the number of inputs
     * @param nOut the number of outputs
     * @param activation the activation function for the layer
     * @param rng the rng to use for sampling
     * @param layerInput the layer starting input
     * @param dist the probability distribution to use
     * for generating weights
     * @return a hidden layer with the given paremters
     */
    public HiddenLayer createHiddenLayer(int index,int nIn,int nOut,ActivationFunction activation,RandomGenerator rng,DoubleMatrix layerInput,RealDistribution dist) {
        return super.createHiddenLayer(index, nIn, nOut, activation, rng, layerInput, dist);

    }


    @Override
    public void pretrain(DoubleMatrix input, Object[] otherParams) {
        int k = (Integer) otherParams[0];
        double lr = (Double) otherParams[1];
        int epochs = (Integer) otherParams[2];
        pretrain(input,k,lr,epochs);

    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     * @param input the input to train on
     * @param k the k to use for running the RBM contrastive divergence.
     * The typical tip is that the higher k is the closer to the model
     * you will be approximating due to more sampling. K = 1
     * usually gives very good results and is the default in quite a few situations.
     * @param learningRate the learning rate to use
     * @param epochs the number of epochs to train
     */
    public void pretrain(DoubleMatrix input,int k,double learningRate,int epochs) {
        /*During pretrain, feed forward expected activations of network, use activation functions during pretrain  */
        if(this.getInput() == null || this.getLayers() == null || this.getLayers()[0] == null || this.getSigmoidLayers() == null || this.getSigmoidLayers()[0] == null) {
            setInput(input);
            initializeLayers(input);
        }
        else
            setInput(input);

        DoubleMatrix layerInput = null;

        for(int i = 0; i < getnLayers(); i++) {
            if(i == 0)
                layerInput = getInput();
            else {
                if(isUseHiddenActivationsForwardProp())
                    layerInput = getSigmoidLayers()[i - 1].sampleHGivenV(layerInput);
                else
                    layerInput = getLayers()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();

            }
            log.info("Training on layer " + (i + 1));
            //override learning rate where present
            double realLearningRate = layerLearningRates.get(i) != null ? layerLearningRates.get(i) : learningRate;
            if(isForceNumEpochs()) {
                for(int epoch = 0; epoch < epochs; epoch++) {
                    log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
                    getLayers()[i].train(layerInput, realLearningRate,new Object[]{k,learningRate});
                    getLayers()[i].epochDone(epoch);
                }
            }
            else
                getLayers()[i].trainTillConvergence(layerInput, realLearningRate, new Object[]{k,realLearningRate,epochs});


        }
    }

    public Map<Integer, RBM.VisibleUnit> getVisibleUnitByLayer() {
        return visibleUnitByLayer;
    }

    public void setVisibleUnitByLayer(Map<Integer, RBM.VisibleUnit> visibleUnitByLayer) {
        this.visibleUnitByLayer = visibleUnitByLayer;
    }

    public Map<Integer, RBM.HiddenUnit> getHiddenUnitByLayer() {
        return hiddenUnitByLayer;
    }

    public void setHiddenUnitByLayer(Map<Integer, RBM.HiddenUnit> hiddenUnitByLayer) {
        this.hiddenUnitByLayer = hiddenUnitByLayer;
    }

    public RBM.VisibleUnit getVisibleUnit() {
        return visibleUnit;
    }

    public void setVisibleUnit(RBM.VisibleUnit visibleUnit) {
        this.visibleUnit = visibleUnit;
    }

    public RBM.HiddenUnit getHiddenUnit() {
        return hiddenUnit;
    }

    public void setHiddenUnit(RBM.HiddenUnit hiddenUnit) {
        this.hiddenUnit = hiddenUnit;
    }

    public void pretrain(int k,double learningRate,int epochs) {
        pretrain(this.getInput(),k,learningRate,epochs);
    }


    @Override
    public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
                                     int nHidden, DoubleMatrix W, DoubleMatrix hBias,
                                     DoubleMatrix vBias, RandomGenerator rng,int index) {

        RBM ret = new RBM.Builder()
                .withHidden(hiddenUnitByLayer.get(index) != null ? hiddenUnitByLayer.get(index) : hiddenUnit).withVisible(visibleUnitByLayer.get(index) != null ? visibleUnitByLayer.get(index) : visibleUnit)
                .useRegularization(isUseRegularization()).withOptmizationAlgo(getOptimizationAlgorithm()).withL2(getL2())
                .useAdaGrad(isUseAdaGrad()).normalizeByInputRows(isNormalizeByInputRows()).withLossFunction(getLossFunction())
                .withMomentum(getMomentum()).withSparsity(getSparsity()).withDistribution(getDist()).normalizeByInputRows(normalizeByInputRows)
                .numberOfVisible(nVisible).numHidden(nHidden).withWeights(W).withDropOut(dropOut)
                .withInput(input).withVisibleBias(vBias).withHBias(hBias).withDistribution(getDist())
                .withRandom(rng).renderWeights(getRenderWeightsEveryNEpochs())
                .fanIn(getFanIn()).build();

        return ret;
    }



    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new RBM[numLayers];
    }


    public static class Builder extends BaseMultiLayerNetwork.Builder<DBN> {

        private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
        private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
        private Map<Integer,RBM.VisibleUnit> visibleUnitByLayer = new HashMap<>();
        private Map<Integer,RBM.HiddenUnit> hiddenUnitByLayer = new HashMap<>();


        public Builder() {
            this.clazz = DBN.class;
        }


        public Builder withVisibleUnitsByLayer(Map<Integer,RBM.VisibleUnit> visibleUnitByLayer) {
            this.visibleUnitByLayer.putAll(visibleUnitByLayer);
            return this;
        }

        public Builder withHiddenUnitsByLayer(Map<Integer,RBM.HiddenUnit> hiddenUnitByLayer) {
            this.hiddenUnitByLayer.putAll(hiddenUnitByLayer);
            return this;
        }

        public Builder withVisibleUnits(RBM.VisibleUnit visibleUnit) {
            this.visibleUnit = visibleUnit;
            return this;
        }


        public Builder withHiddenUnits(RBM.HiddenUnit hiddenUnit) {
            this.hiddenUnit = hiddenUnit;
            return this;
        }



        public Builder activateForLayer(Map<Integer,ActivationFunction> activationForLayer) {
            super.activateForLayer(activationForLayer);
            return this;
        }

        public Builder activateForLayer(int layer,ActivationFunction function) {
            super.activateForLayer(layer,function);
            return this;
        }

        /**
         * Activation function for output layer
         * @param outputActivationFunction the output activation function to use
         * @return builder pattern
         */
        public Builder withOutputActivationFunction(ActivationFunction outputActivationFunction) {
            super.withOutputActivationFunction(outputActivationFunction);
            return this;
        }


        /**
         * Output loss function
         * @param outputLossFunction the output loss function
         * @return
         */
        public Builder withOutputLossFunction(OutputLayer.LossFunction outputLossFunction) {
            super.withOutputLossFunction(outputLossFunction);
            return this;
        }

        /**
         * Which optimization algorithm to use with neural nets and Logistic regression
         * @param optimizationAlgo which optimization algorithm to use with
         * neural nets and logistic regression
         * @return builder pattern
         */
        public Builder withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
            super.withOptimizationAlgorithm(optimizationAlgo);
            return this;
        }

        /**
         * Loss function to use with neural networks
         * @param lossFunction loss function to use with neural networks
         * @return builder pattern
         */
        public Builder withLossFunction(NeuralNetwork.LossFunction lossFunction) {
            super.withLossFunction(lossFunction);
            return this;
        }

        /**
         * Whether to use drop out on the neural networks or not:
         * random zero out of examples
         * @param dropOut the dropout to use
         * @return builder pattern
         */
        public Builder withDropOut(double dropOut) {
            super.withDropOut(dropOut);
            return this;
        }

        /**
         * Whether to use hidden layer activations or neural network sampling
         * on feed forward pass
         * @param useHiddenActivationsForwardProp true if use hidden activations, false otherwise
         * @return builder pattern
         */
        public Builder useHiddenActivationsForwardProp(boolean useHiddenActivationsForwardProp) {
            super.useHiddenActivationsForwardProp(useHiddenActivationsForwardProp);
            return this;
        }

        /**
         * Turn this off for full dataset training
         * @param normalizeByInputRows whether to normalize the changes
         * by the number of input rows
         * @return builder pattern
         */
        public Builder normalizeByInputRows(boolean normalizeByInputRows) {
            super.normalizeByInputRows(normalizeByInputRows);
            return this;
        }




        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        public Builder withSparsity(double sparsity) {
            super.withSparsity(sparsity);
            return this;
        }


        public Builder withVisibleBiasTransforms(Map<Integer,MatrixTransform> visibleBiasTransforms) {
            super.withVisibleBiasTransforms(visibleBiasTransforms);
            return this;
        }

        public Builder withHiddenBiasTransforms(Map<Integer,MatrixTransform> hiddenBiasTransforms) {
            super.withHiddenBiasTransforms(hiddenBiasTransforms);
            return this;
        }

        /**
         * Forces use of number of epochs for training
         * SGD style rather than conjugate gradient
         * @return
         */
        public Builder forceEpochs() {
            shouldForceEpochs = true;
            return this;
        }

        /**
         * Disables back propagation
         * @return
         */
        public Builder disableBackProp() {
            backProp = false;
            return this;
        }

        /**
         * Transform the weights at the given layer
         * @param layer the layer to transform
         * @param transform the function used for transformation
         * @return
         */
        public Builder transformWeightsAt(int layer,MatrixTransform transform) {
            weightTransforms.put(layer,transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given layers
         * @param transforms
         * @return
         */
        public Builder transformWeightsAt(Map<Integer,MatrixTransform> transforms) {
            weightTransforms.putAll(transforms);
            return this;
        }

        /**
         * Probability distribution for generating weights
         * @param dist
         * @return
         */
        public Builder withDist(RealDistribution dist) {
            super.withDist(dist);
            return this;
        }

        /**
         * Specify momentum
         * @param momentum
         * @return
         */
        public Builder withMomentum(double momentum) {
            super.withMomentum(momentum);
            return this;
        }

        /**
         * Use l2 reg
         * @param useRegularization
         * @return
         */
        public Builder useRegularization(boolean useRegularization) {
            super.useRegularization(useRegularization);
            return this;
        }

        /**
         * L2 coefficient
         * @param l2
         * @return
         */
        public Builder withL2(double l2) {
            super.withL2(l2);
            return this;
        }

        /**
         * Whether to plot weights or not
         * @param everyN
         * @return
         */
        public Builder renderWeights(int everyN) {
            super.renderWeights(everyN);
            return this;
        }

        public Builder withFanIn(Double fanIn) {
            super.withFanIn(fanIn);
            return this;
        }

        /**
         * Pick an activation function, default is sigmoid
         * @param activation
         * @return
         */
        public Builder withActivation(ActivationFunction activation) {
            super.withActivation(activation);
            return this;
        }


        public Builder numberOfInputs(int nIns) {
            super.numberOfInputs(nIns);
            return this;
        }


        public Builder hiddenLayerSizes(Integer[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }

        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }

        public Builder numberOfOutPuts(int nOuts) {
            super.numberOfOutPuts(nOuts);
            return this;
        }

        public Builder withRng(RandomGenerator gen) {
            super.withRng(gen);
            return this;
        }

        public Builder withInput(DoubleMatrix input) {
            super.withInput(input);
            return this;
        }

        public Builder withLabels(DoubleMatrix labels) {
            super.withLabels(labels);
            return this;
        }

        public Builder withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz =  clazz;
            return this;
        }



        public DBN build() {
            DBN ret = super.build();
            ret.hiddenUnit = hiddenUnit;
            ret.visibleUnit = visibleUnit;
            ret.visibleUnitByLayer.putAll(visibleUnitByLayer);
            ret.hiddenUnitByLayer.putAll(hiddenUnitByLayer);
            return ret;
        }
    }




}
