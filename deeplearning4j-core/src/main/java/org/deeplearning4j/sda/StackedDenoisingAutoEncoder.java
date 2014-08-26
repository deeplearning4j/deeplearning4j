package org.deeplearning4j.sda;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.WeightInit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


/**
 *  Stacked Denoising AutoEncoders are merely denoising autoencoders
 *  who's inputs feed in to the next one.
 * @author Adam Gibson
 *
 */
public class StackedDenoisingAutoEncoder extends BaseMultiLayerNetwork  {

    private static final long serialVersionUID = 1448581794985193009L;
    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoder.class);



    private StackedDenoisingAutoEncoder() {}

    private StackedDenoisingAutoEncoder(int n_ins, int[] hiddenLayerSizes, int nOuts,
                                        int nLayers, RandomGenerator rng, INDArray input,INDArray labels) {
        super(n_ins, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);

    }


    private StackedDenoisingAutoEncoder(int nIns, int[] hiddenLayerSizes, int nOuts,
                                        int n_layers, RandomGenerator rng) {
        super(nIns, hiddenLayerSizes, nOuts, n_layers, rng);
    }


    public void pretrain( float lr,  float corruptionLevel,  int epochs) {
        pretrain(this.getInput(),lr,corruptionLevel,epochs);
    }

    /**
     * Pretrain with a data applyTransformToDestination iterator.
     * This will run through each neural net at a time and train on the input.
     *
     * @param iter        the iterator to use
     * @param otherParams
     */
    @Override
    public void pretrain(DataSetIterator iter, Object[] otherParams) {
        float corruptionLevel = (float) otherParams[0];
        float lr = (Float) otherParams[1];
        int epochs = (Integer) otherParams[2];
        int passes = otherParams.length > 3 ? (Integer) otherParams[3] : 1;
        for(int i = 0; i < passes; i++)
            pretrain(iter,corruptionLevel,lr,epochs);

    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     * @param iter the input to train on
     * @param corruptionLevel the corruption level to use for running the denoising autoencoder training.
     * The typical tip is that the higher k is the closer to the model
     * you will be approximating due to more sampling. K = 1
     * usually gives very good results and is the default in quite a few situations.
     * @param lr the learning rate to use
     * @param iterations the number of epochs to train
     */
    public void pretrain(DataSetIterator iter,float corruptionLevel,float lr,int iterations) {

        INDArray layerInput;

        for (int i = 0; i < getnLayers(); i++) {
            if (i == 0) {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    this.input = next.getFeatureMatrix();
                      /*During pretrain, feed forward expected activations of network, use activation functions during pretrain  */
                    if(this.getInput() == null || this.getLayers() == null || this.getLayers()[0] == null || this.getSigmoidLayers() == null || this.getSigmoidLayers()[0] == null) {
                        setInput(input);
                        initializeLayers(input);
                    }
                    else
                        setInput(input);
                    //override learning rate where present
                    float realLearningRate = layerLearningRates.get(i) != null ? layerLearningRates.get(i) : lr;
                    if (isForceNumEpochs()) {
                        for (int iteration = 0; iteration < iterations; iteration++) {
                            log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
                            getLayers()[i].train(next.getFeatureMatrix(), realLearningRate, new Object[]{corruptionLevel, lr});
                            getLayers()[i].iterationDone(iteration);
                        }
                    } else
                        getLayers()[i].trainTillConvergence(next.getFeatureMatrix(), realLearningRate, new Object[]{corruptionLevel, realLearningRate, iterations});

                }

                iter.reset();
            }



            else {
                boolean activateOnly = getSampleOrActivate() != null && getSampleOrActivate().get(i) != null ? getSampleOrActivate().get(i) : !sampleFromHiddenActivations;
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    layerInput = next.getFeatureMatrix();
                    for(int j = 1; j <= i; j++) {
                        if(activateOnly)
                            layerInput = getSigmoidLayers()[j - 1].activate(layerInput);
                        else if(isSampleFromHiddenActivations())
                            layerInput = getLayers()[j - 1].sampleHiddenGivenVisible(getSigmoidLayers()[j - 1].getActivationFunction().apply(layerInput)).getSecond();
                        else
                            layerInput = getLayers()[j - 1].sampleHiddenGivenVisible(layerInput).getSecond();

                    }


                    log.info("Training on layer " + (i + 1));
                    //override learning rate where present
                    float realLearningRate = layerLearningRates.get(i) != null ? layerLearningRates.get(i) : lr;
                    if(isForceNumEpochs()) {
                        for(int iteration = 0; iteration < iterations; iteration++) {
                            log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
                            getLayers()[i].train(layerInput, realLearningRate,new Object[]{corruptionLevel,lr});
                            getLayers()[i].iterationDone(iteration);
                        }
                    }
                    else
                        getLayers()[i].trainTillConvergence(layerInput, realLearningRate, new Object[]{corruptionLevel,realLearningRate,iterations});



                }

                iter.reset();

            }
        }

    }

    @Override
    public void pretrain(INDArray input, Object[] otherParams) {
        if(otherParams == null) {
            otherParams = new Object[]{0.01,0.3,1000};
        }

        Float lr = (Float) otherParams[0];
        Float corruptionLevel = (Float) otherParams[1];
        Integer iterations = (Integer) otherParams[2];

        pretrain(input, lr, corruptionLevel, iterations);

    }

    /**
     * Unsupervised pretraining based on reconstructing the input
     * from a corrupted version
     * @param input the input to train on
     * @param lr the starting learning rate
     * @param corruptionLevel the corruption level (the smaller number of inputs; the higher the
     * corruption level should be) the percent of inputs to corrupt
     * @param iterations the number of iterations to run
     */
    public void pretrain(INDArray input,float lr,  float corruptionLevel,  int iterations) {
        if(this.getInput() == null)
            initializeLayers(input.dup());

        if(isUseGaussNewtonVectorProductBackProp())
            log.warn("Warning; using gauss newton vector back prop with pretrain is known to cause issues with obscenely large activations.");

        this.input = input;

        INDArray layerInput = null;

        for(int i = 0; i < this.getnLayers(); i++) {  // layer-wise
            //input layer
            if(i == 0)
                layerInput = input;
            else
                layerInput = this.getLayers()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();
            if(isForceNumEpochs()) {
                for(int iteration = 0; iteration < iterations; iteration++) {
                    getLayers()[i].train(layerInput, lr,  new Object[]{corruptionLevel,lr});
                    log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
                    getLayers()[i].iterationDone(iteration);

                }
            }
            else
                getLayers()[i].trainTillConvergence(layerInput, lr, new Object[]{corruptionLevel,lr,iterations});


        }
    }


    @Override
    public NeuralNetwork createLayer(INDArray input, int nVisible,
                                     int nHidden, INDArray W, INDArray hbias,
                                     INDArray vBias, RandomGenerator rng,int index) {
        DenoisingAutoEncoder ret = new DenoisingAutoEncoder.Builder()
                .withDropOut(dropOut).constrainGradientToUnitNorm(constrainGradientToUnitNorm).weightInit(weightInitByLayer.get(index) != null ? weightInitByLayer.get(index) : weightInit)
                .withLossFunction(lossFunctionByLayer.get(index) != null ? lossFunctionByLayer.get(index) : getLossFunction())
                .withHBias(hbias).withInput(input).withWeights(W).withDistribution(getDist()).withOptmizationAlgo(getOptimizationAlgorithm())
                .withRandom(rng).withMomentum(getMomentum()).withVisibleBias(vBias).normalizeByInputRows(normalizeByInputRows)
                .numberOfVisible(nVisible).numHidden(nHidden).withDistribution(getDist())
                .momentumAfter(momentumAfterByLayer.get(index) != null ? momentumAfterByLayer.get(index) : momentumAfter)
                .resetAdaGradIterations(resetAdaGradIterationsByLayer.get(index) != null ? resetAdaGradIterationsByLayer.get(index) : resetAdaGradIterations)
                .withSparsity(getSparsity()).renderWeights(renderByLayer.get(index) != null ? renderByLayer.get(index) : renderWeightsEveryNEpochs)
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
         * Output layer weight initialization
         *
         * @param outputLayerWeightInit
         * @return
         */
        @Override
        public Builder outputLayerWeightInit(WeightInit outputLayerWeightInit) {
            super.outputLayerWeightInit(outputLayerWeightInit);
            return this;
        }

        /**
         * Layer specific weight init
         *
         * @param weightInitByLayer
         * @return
         */
        @Override
        public Builder weightInitByLayer(Map<Integer, WeightInit> weightInitByLayer) {
            super.weightInitByLayer(weightInitByLayer);
            return this;
        }

        /**
         * Default weight init scheme
         *
         * @param weightInit
         * @return
         */
        @Override
        public Builder weightInit(WeightInit weightInit) {
            super.weightInit(weightInit);
            return this;
        }



        /**
         * Whether to constrain gradient to unit norm or not
         * @param constrainGradientToUnitNorm
         * @return
         */
        public Builder constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            super.constrainGradientToUnitNorm(constrainGradientToUnitNorm);
            return this;
        }

        public Builder useGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
            super.useGaussNewtonVectorProductBackProp(useGaussNewtonVectorProductBackProp);
            return this;
        }


        /**
         * Use drop connect on activations or not
         *
         * @param useDropConnect use drop connect or not
         * @return builder pattern
         */
        @Override
        public  Builder useDropConnection(boolean useDropConnect) {
            super.useDropConnection(useDropConnect);
            return this;
        }

        /**
         * Output layer drop out
         *
         * @param outputLayerDropout
         * @return
         */
        @Override
        public Builder outputLayerDropout(float outputLayerDropout) {
            super.outputLayerDropout(outputLayerDropout);
            return this;
        }

        @Override
        public Builder lineSearchBackProp(boolean lineSearchBackProp) {
            super.lineSearchBackProp(lineSearchBackProp);
            return this;
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
        public Builder learningRateForLayer(Map<Integer, Float> learningRates) {
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
        public Builder withDropOut(float dropOut) {
            super.withDropOut(dropOut);
            return this;
        }

        /**
         * Whether to use hidden layer activations or neural network sampling
         * on feed forward pass
         * @param useHiddenActivationsForwardProp true if use hidden activations, false otherwise
         * @return builder pattern
         */
        public Builder sampleFromHiddenActivations(boolean useHiddenActivationsForwardProp) {
            super.sampleFromHiddenActivations(useHiddenActivationsForwardProp);
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

        public Builder withSparsity(float sparsity) {
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
         * @param layer the layer to applyTransformToOrigin
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
        public Builder withMomentum(float momentum) {
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
        public Builder withL2(float l2) {
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

        public Builder withInput(INDArray input) {
            super.withInput(input);
            return this;
        }


        /**
         * Reset the adagrad epochs  after n iterations
         *
         * @param resetAdaGradIterations the number of iterations to reset adagrad after
         * @return
         */
        @Override
        public  Builder resetAdaGradIterations(int resetAdaGradIterations) {
            super.resetAdaGradIterations(resetAdaGradIterations);
            return this;
        }

        /**
         * Reset map for adagrad historical gradient by layer
         *
         * @param resetAdaGradEpochsByLayer
         * @return
         */
        @Override
        public Builder resetAdaGradEpochsByLayer(Map<Integer, Integer> resetAdaGradEpochsByLayer) {
            super.resetAdaGradEpochsByLayer(resetAdaGradEpochsByLayer);
            return this;
        }

        /**
         * Sets the momentum to the specified value for a given layer after a specified number of iterations
         *
         * @param momentumAfterByLayer the by layer momentum changes
         * @return
         */
        @Override
        public Builder momentumAfterByLayer(Map<Integer, Map<Integer, Float>> momentumAfterByLayer) {
            super.momentumAfterByLayer(momentumAfterByLayer);
            return this;
        }

        /**
         * Set the momentum to the value after the desired number of iterations
         *
         * @param momentumAfter the momentum after n iterations
         * @return
         */
        @Override
        public Builder momentumAfter(Map<Integer, Float> momentumAfter) {
            super.momentumAfter(momentumAfter);
            return this;
        }

        public Builder withLabels(INDArray labels) {
            super.withLabels(labels);
            return this;
        }

        public Builder withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz =  clazz;
            return this;
        }

    }








}