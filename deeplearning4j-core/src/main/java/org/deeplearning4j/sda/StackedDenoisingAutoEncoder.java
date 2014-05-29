package org.deeplearning4j.sda;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.transformation.MatrixTransform;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


/**
 * A JBlas implementation of 
 * stacked denoising auto encoders.
 * @author Adam Gibson
 *
 */
public class StackedDenoisingAutoEncoder extends BaseMultiLayerNetwork  {

    private static final long serialVersionUID = 1448581794985193009L;
    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoder.class);



    private StackedDenoisingAutoEncoder() {}

    private StackedDenoisingAutoEncoder(int n_ins, int[] hiddenLayerSizes, int nOuts,
                                        int nLayers, RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
        super(n_ins, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);

    }


    private StackedDenoisingAutoEncoder(int nIns, int[] hiddenLayerSizes, int nOuts,
                                        int n_layers, RandomGenerator rng) {
        super(nIns, hiddenLayerSizes, nOuts, n_layers, rng);
    }


    public void pretrain( double lr,  double corruptionLevel,  int epochs) {
        pretrain(this.getInput(),lr,corruptionLevel,epochs);
    }


    @Override
    public void pretrain(DoubleMatrix input, Object[] otherParams) {
        if(otherParams == null) {
            otherParams = new Object[]{0.01,0.3,1000};
        }

        Double lr = (Double) otherParams[0];
        Double corruptionLevel = (Double) otherParams[1];
        Integer epochs = (Integer) otherParams[2];

        pretrain(input, lr, corruptionLevel, epochs);

    }

    /**
     * Unsupervised pretraining based on reconstructing the input
     * from a corrupted version
     * @param input the input to train on
     * @param lr the starting learning rate
     * @param corruptionLevel the corruption level (the smaller number of inputs; the higher the
     * corruption level should be) the percent of inputs to corrupt
     * @param epochs the number of iterations to run
     */
    public void pretrain(DoubleMatrix input,double lr,  double corruptionLevel,  int epochs) {
        if(this.getInput() == null)
            initializeLayers(input.dup());

        DoubleMatrix layerInput = null;

        for(int i = 0; i < this.getnLayers(); i++) {  // layer-wise
            //input layer
            if(i == 0)
                layerInput = input;
            else
                layerInput = this.getSigmoidLayers()[i - 1].sampleHGivenV(layerInput);
            if(isForceNumEpochs()) {
                for(int epoch = 0; epoch < epochs; epoch++) {
                    getLayers()[i].train(layerInput, lr,  new Object[]{corruptionLevel,lr});
                    log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
                    getLayers()[i].epochDone(epoch);

                }
            }
            else
                getLayers()[i].trainTillConvergence(layerInput, lr, new Object[]{corruptionLevel,lr,epochs});


        }
    }

    /**
     *
     * @param input input examples
     * @param labels output labels
     * @param otherParams
     *
     * (double) learningRate
     * (double) corruptionLevel
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
        if(otherParams == null) {
            otherParams = new Object[]{0.01,0.3,1000};
        }

        Double lr = (Double) otherParams[0];
        Double corruptionLevel = (Double) otherParams[1];
        Integer epochs = (Integer) otherParams[2];

        pretrain(input, lr, corruptionLevel, epochs);
        if(otherParams.length <= 3)
            finetune(labels, lr, epochs);
        else {
            Double finetuneLr = (Double) otherParams[3];
            Integer fineTuneEpochs = (Integer) otherParams[4];
            finetune(labels,finetuneLr,fineTuneEpochs);
        }
    }



    @Override
    public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
                                     int nHidden, DoubleMatrix W, DoubleMatrix hbias,
                                     DoubleMatrix vBias, RandomGenerator rng,int index) {
        DenoisingAutoEncoder ret = new DenoisingAutoEncoder.Builder().withDropOut(dropOut)
                .withLossFunction(lossFunctionByLayer.get(index) != null ? lossFunctionByLayer.get(index) :  getLossFunction())
                .withHBias(hbias).withInput(input).withWeights(W).withDistribution(getDist()).withOptmizationAlgo(getOptimizationAlgorithm())
                .withRandom(rng).withMomentum(getMomentum()).withVisibleBias(vBias).normalizeByInputRows(normalizeByInputRows)
                .numberOfVisible(nVisible).numHidden(nHidden).withDistribution(getDist())
                .withSparsity(getSparsity()).renderWeights(renderByLayer.get(index) != null ? renderByLayer.get(index) : renderWeightsEveryNEpochs).fanIn(getFanIn())
                .build();
        return ret;
    }


    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new DenoisingAutoEncoder[numLayers];
    }


    public static class Builder extends BaseMultiLayerNetwork.Builder<StackedDenoisingAutoEncoder> {
        public Builder() {
            this.clazz = StackedDenoisingAutoEncoder.class;
        }
        /**
         * Sample or activate by layer allows for deciding to sample or just pass straight activations
         * for each layer
         *
         * @param sampleOrActivateByLayer
         * @return
         */
        @Override
        public Builder sampleOrActivateByLayer(Map<Integer, Boolean> sampleOrActivateByLayer) {
            super.sampleOrActivateByLayer(sampleOrActivateByLayer);
            return this;
        }

        @Override
        public Builder renderByLayer(Map<Integer, Integer> renderByLayer) {
            super.renderByLayer(renderByLayer);
            return this;
        }

        @Override
        public Builder learningRateForLayer(Map<Integer, Double> learningRates) {
            super.learningRateForLayer(learningRates);
            return this;
        }

        /**
         * Loss function by layer
         *
         * @param lossFunctionByLayer the loss function per layer
         * @return builder pattern
         */
        @Override
        public Builder lossFunctionByLayer(Map<Integer, NeuralNetwork.LossFunction> lossFunctionByLayer) {
            super.lossFunctionByLayer(lossFunctionByLayer);
            return this;
        }

        @Override
        public Builder activateForLayer(Map<Integer,ActivationFunction> activationForLayer) {
            super.activateForLayer(activationForLayer);
            return this;
        }

        @Override
        public Builder activateForLayer(int layer,ActivationFunction function) {
            super.activateForLayer(layer,function);
            return this;
        }

        /**
         * Activation function for output layer
         * @param outputActivationFunction the output activation function to use
         * @return builder pattern
         */
        @Override
        public Builder withOutputActivationFunction(ActivationFunction outputActivationFunction) {
            super.withOutputActivationFunction(outputActivationFunction);
            return this;
        }


        /**
         * Output loss function
         * @param outputLossFunction the output loss function
         * @return
         */
        @Override
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
        @Override
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

    }








}