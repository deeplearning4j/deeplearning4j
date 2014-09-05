package org.deeplearning4j.models.classifiers.dbn;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.*;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.Layer;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


/**
 * Deep Belief Network. This is a MultiLayer Perceptron Model
 * using Restricted Boltzmann Machines.
 *  See Hinton's practical guide to RBMs for great examples on
 *  how to iterate and tune parameters.
 *
 * @author Adam Gibson
 *
 */
public class DBN extends BaseMultiLayerNetwork {

    private static final long serialVersionUID = -9068772752220902983L;
    private static Logger log = LoggerFactory.getLogger(DBN.class);
    private boolean useRBMPropUpAsActivations = true;

    public DBN() {}







    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overridden
     * for other kinds of hidden units
     * @param layerInput the layer starting input
     * for generating weights
     * @return a hidden layer with the given parameters
     */
    public org.deeplearning4j.nn.layers.Layer createHiddenLayer(int index,INDArray layerInput) {
        return (Layer) super.createHiddenLayer(index,layerInput);

    }

    @Override
    public org.deeplearning4j.nn.api.Layer createHiddenLayer(int index, int nIn, int nOut, INDArray layerInput) {
        return null;
    }


    @Override
    public void pretrain(DataSetIterator iter, Object[] otherParams) {
        int passes = otherParams.length > 3 ? (Integer) otherParams[3] : 1;
        for(int i = 0; i < passes; i++)
            pretrain(input, defaultConfiguration.getK(),defaultConfiguration.getLr(),defaultConfiguration.getNumIterations());


    }

    @Override
    public void pretrain(INDArray input, Object[] otherParams) {
        pretrain(input, defaultConfiguration.getK(),defaultConfiguration.getLr(),defaultConfiguration.getNumIterations());

    }

    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     * @param iter the input to iterate on
     * @param k the k to use for running the RBM contrastive divergence.
     * The typical tip is that the higher k is the closer to the model
     * you will be approximating due to more sampling. K = 1
     * usually gives very good results and is the default in quite a few situations.
     * @param learningRate the learning rate to use
     * @param epochs the number of epochs to iterate
     */
    public void pretrain(DataSetIterator iter,int k,float learningRate,int epochs) {


        INDArray layerInput;

        for(int i = 0; i < getnLayers(); i++) {
            if(i == 0) {
                while(iter.hasNext()) {
                    DataSet next = iter.next();
                    this.input = next.getFeatureMatrix();
                      /*During pretrain, feed forward expected activations of network, use activation functions during pretrain  */
                    if(this.getInput() == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null) {
                        setInput(input);
                        initializeLayers(input);
                    }
                    else
                        setInput(input);
                    //override learning rate where present
                    float realLearningRate = layerWiseConfigurations.get(i).getLr();
                    if(isForceNumEpochs()) {
                        for(int epoch = 0; epoch < epochs; epoch++) {
                            log.info("Error on iteration " + epoch + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                            getNeuralNets()[i].iterate(next.getFeatureMatrix(), new Object[]{k, learningRate});
                            getNeuralNets()[i].iterationDone(epoch);
                        }
                    }
                    else
                        getNeuralNets()[i].fit(next.getFeatureMatrix(), new Object[]{k, realLearningRate, epochs});

                }

                iter.reset();
            }

            else {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    layerInput = next.getFeatureMatrix();
                    for(int j = 1; j <= i; j++)
                        layerInput = activationFromPrevLayer(j,layerInput);




                    log.info("Training on layer " + (i + 1));
                    //override learning rate where present
                    float realLearningRate = layerWiseConfigurations.get(i).getLr();
                    if(isForceNumEpochs()) {
                        for(int epoch = 0; epoch < epochs; epoch++) {
                            log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                            getNeuralNets()[i].iterate(layerInput, new Object[]{k, learningRate});
                            getNeuralNets()[i].iterationDone(epoch);
                        }
                    }
                    else
                        getNeuralNets()[i].fit(layerInput, new Object[]{k, realLearningRate, epochs});

                }

                iter.reset();


            }
        }
    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     * @param input the input to iterate on
     * @param k the k to use for running the RBM contrastive divergence.
     * The typical tip is that the higher k is the closer to the model
     * you will be approximating due to more sampling. K = 1
     * usually gives very good results and is the default in quite a few situations.
     * @param learningRate the learning rate to use
     * @param epochs the number of epochs to iterate
     */
    public void pretrain(INDArray input,int k, float learningRate,int epochs) {

        if(isUseGaussNewtonVectorProductBackProp()) {
            log.warn("WARNING; Gauss newton back vector back propagation is primarily used for hessian free which does not involve pretrain; just finetune. Use this at your own risk");
        }


        /*During pretrain, feed forward expected activations of network, use activation functions during pretrain  */
        if(this.getInput() == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null) {
            setInput(input);
            initializeLayers(input);
        }
        else
            setInput(input);

        INDArray layerInput = null;

        for(int i = 0; i < getnLayers(); i++) {
            if(i == 0)
                layerInput = getInput();
            else
                layerInput = activationFromPrevLayer(i -1,layerInput);


            log.info("Training on layer " + (i + 1));
            //override learning rate where present
            float realLearningRate = layers[i].conf().getLr();
            if(isForceNumEpochs()) {
                for(int epoch = 0; epoch < epochs; epoch++) {
                    log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                    getNeuralNets()[i].iterate(layerInput, new Object[]{k, learningRate});
                    getNeuralNets()[i].iterationDone(epoch);
                }
            }
            else
                getNeuralNets()[i].fit(layerInput, new Object[]{k, realLearningRate, epochs});


        }
    }




    public void pretrain(int k,float learningRate,int epochs) {
        pretrain(this.getInput(),k,learningRate,epochs);
    }


    @Override
    public NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hBias,
                                     INDArray vBias, int index) {
        RBM ret = new RBM.Builder().withInput(input)
                .withWeights(W).withHBias(hBias).withVisibleBias(vBias)
                .configure(layerWiseConfigurations.get(index))
                .build();

        return ret;
    }



    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new RBM[numLayers];
    }

    /**
     * Transform the data based on the model's output.
     * This can be anything from a number to reconstructions.
     *
     * @param data the data to transform
     * @return the transformed data
     */
    @Override
    public INDArray transform(INDArray data) {
        return output(data);
    }

    /**
     * Fit the model to the given data
     *
     * @param data   the data to fit the model to
     * @param params the params (mixed values)
     */
    @Override
    public void fit(INDArray data, Object[] params) {
        pretrain(data,defaultConfiguration.getK(),defaultConfiguration.getLr(),defaultConfiguration.getNumIterations());
    }


    public static class Builder extends BaseMultiLayerNetwork.Builder<DBN> {
        private boolean useRBMPropUpAsActivation = false;

        public Builder() {
            this.clazz = DBN.class;
        }


        @Override
        public Builder configure(NeuralNetConfiguration conf) {
            super.configure(conf);
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




        @Override
        public Builder lineSearchBackProp(boolean lineSearchBackProp) {
            super.lineSearchBackProp(lineSearchBackProp);
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
         * the given neuralNets
         * @param transforms
         * @return
         */
        public Builder transformWeightsAt(Map<Integer,MatrixTransform> transforms) {
            weightTransforms.putAll(transforms);
            return this;
        }

        @Override
        public Builder hiddenLayerSizes(Integer[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }

        @Override
        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }



        @Override
        public Builder withInput(INDArray input) {
            super.withInput(input);
            return this;
        }

        @Override
        public Builder withLabels(INDArray labels) {
            super.withLabels(labels);
            return this;
        }

        @Override
        public Builder withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz =  clazz;
            return this;
        }



        @Override
        public DBN build() {
            DBN ret = super.build();
            ret.useRBMPropUpAsActivations = useRBMPropUpAsActivation;
            ret.initializeLayers(Nd4j.zeros(1, ret.defaultConfiguration.getnIn()));
            return ret;
        }
    }




}
