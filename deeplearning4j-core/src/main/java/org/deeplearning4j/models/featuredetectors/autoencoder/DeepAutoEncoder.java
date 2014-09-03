package org.deeplearning4j.models.featuredetectors.autoencoder;


import java.util.*;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.api.DataSet;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.*;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.layers.Layer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;




/**
 * Encapsulates a deep auto encoder and decoder (the transpose of an encoder).
 *
 * The idea here will be to iterate a DBN first using pretrain. This creates the weights
 * for the autoencoder. Here is an example configuration for training a DBN for binary images.
 *
 *


 Map<Integer,Double> layerLearningRates = new HashMap<>();
 layerLearningRates.put(codeLayer,1e-1);
 RandomGenerator rng = new MersenneTwister(123);


 DBN dbn = new DBN.Builder()
 .learningRateForLayer(layerLearningRates)
 .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
 .useRBMPropUpAsActivation(true)
 .activateForLayer(Collections.singletonMap(3, Activations.linear()))
 .withHiddenUnitsByLayer(Collections.singletonMap(codeLayer, RBM.HiddenUnit.GAUSSIAN))
 .numberOfInputs(784)
 .sampleFromHiddenActivations(true)
 .sampleOrActivateByLayer(Collections.singletonMap(3,false))
 .lineSearchBackProp(true).useRegularization(true).withL2(1e-3)
 .withOutputActivationFunction(Activations.sigmoid())
 .numberOfOutPuts(784).withMomentum(0.5)
 .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
 *       .build();
 *
 * Reduction of dimensionality with neural nets Hinton 2006
 * The focus of a deep auto encoder is the code layer.
 * This code layer is the end of the encoder
 * and the input to the decoder.
 *
 * For real valued data, this is a gaussian rectified linear layer.
 *
 * For binary, its binary/binary
 *
 * A few notes from the science 2006 paper:
 * On decode, use straight activations
 * On encode, use sampling from activations
 *
 * The decoder is the transpose of the output layer.
 *

 *
 * Both time should use a loss function that simulates reconstructions:
 * that could be RMSE_XENT or SQUARED_LOSS
 *
 * The code layer should always be gaussian.
 *
 * If the goal is binary codes, the output layer's activation function should be sigmoid.
 *
 *
 *
 * @author Adam Gibson
 *
 *
 */
public class DeepAutoEncoder extends BaseMultiLayerNetwork {

    /**
     *
     */
    private static final long serialVersionUID = -3571832097247806784L;
    private BaseMultiLayerNetwork encoder;
    //learn binary codes
    private ActivationFunction outputLayerActivation = Activations.sigmoid();
    private boolean roundCodeLayerInput = false;
    //could be useful for gaussian/rectified
    private boolean normalizeCodeLayerOutput = false;
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoder.class);

    public DeepAutoEncoder(){}




    /**
     * Pretrain the network with the given parameters
     *
     * @param input       the input to iterate ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void pretrain(INDArray input, Object[] otherParams) {
        throw new IllegalStateException("Not implemented");

    }

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.models.classifiers.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.models.featuredetectors.rbm.RBM} for continuous inputs.
     *
     * @param input    the input to the layer
     * @param W        the weight vector
     * @param hbias    the hidden bias
     * @param vBias    the visible bias
     * @param index    the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.models.featuredetectors.rbm.RBM}
     */
    @Override
    public NeuralNetwork createLayer(INDArray input,INDArray W, INDArray hbias, INDArray vBias,int index) {
        throw new IllegalStateException("Not implemented");
    }


    /* delta computation for back prop with the R operator */
    @Override
    public  List<INDArray> computeDeltasR(INDArray v) {
        List<INDArray> deltaRet = new ArrayList<>();
        INDArray[] deltas = new INDArray[getnLayers() + 1];
        List<INDArray> activations = feedForward();
        List<INDArray> rActivations = feedForwardR(activations,v);
      /*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();
        List<ActivationFunction> activationFunctions = new ArrayList<>();


        for(int j = 0; j < getNeuralNets().length; j++) {
            weights.add(getNeuralNets()[j].getW());
            biases.add(getNeuralNets()[j].gethBias());
            AutoEncoder a = (AutoEncoder) getNeuralNets()[j];
            activationFunctions.add(a.conf().getActivationFunction());
        }

        weights.add(getOutputLayer().getW());
        biases.add(getOutputLayer().getB());
        activationFunctions.add(getOutputLayer().conf().getActivationFunction());

        INDArray rix = rActivations.get(rActivations.size() - 1).div(input.rows());

        //errors
        for(int i = getnLayers(); i >= 0; i--) {
            //W^t * error^l + 1
            deltas[i] =  activations.get(i).transpose().mmul(rix);
            applyDropConnectIfNecessary(deltas[i]);

            if(i > 0)
                rix = rix.mmul(weights.get(i).addRowVector(biases.get(i)).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));




        }

        for(int i = 0; i < deltas.length; i++) {
            if(layerWiseConfigurations.get(i).isConstrainGradientToUnitNorm())
                deltaRet.add(deltas[i].div(deltas[i].norm2(Integer.MAX_VALUE)));

            else
                deltaRet.add(deltas[i]);
        }

        return deltaRet;
    }


    /**
     * Feed forward with the r operator
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    @Override
    public List<INDArray> feedForwardR(List<INDArray> acts,INDArray v) {
        List<INDArray> R = new ArrayList<>();
        R.add(NDArrays.zeros(input.rows(), input.columns()));
        List<Pair<INDArray,INDArray>> vWvB = unPack(v);
        List<INDArray> W = weightMatrices();

        for(int i = 0; i < neuralNets.length; i++) {
            AutoEncoder a = (AutoEncoder) getNeuralNets()[i];
            ActivationFunction derivative = a.conf().getActivationFunction();

            //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
            R.add(R.get(i).mmul(W.get(i)).add(acts.get(i).mmul(vWvB.get(i).getFirst().addRowVector(vWvB.get(i).getSecond())).add(1)).mul((derivative.applyDerivative(acts.get(i + 1)))));
        }

        //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
        R.add(R.get(R.size() - 1).mmul(W.get(W.size() - 1)).add(acts.get(acts.size() - 2).mmul(vWvB.get(vWvB.size() - 1).getFirst().addRowVector(vWvB.get(vWvB.size() - 1).getSecond()))).mul((getOutputLayer().conf().getActivationFunction().applyDerivative(acts.get(acts.size() - 1)))));

        return R;
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
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[numLayers];
    }

    @Override
    public org.deeplearning4j.nn.api.Layer createHiddenLayer(int index, int nIn, int nOut, INDArray layerInput) {
        return null;
    }


    /**
     * Trains the decoder on the given input
     * @param input the given input to iterate on
     */
    public void finetune(INDArray input,float lr,int iterations) {
        this.input = input;


        setInput(input);
        setLabels(input);

        super.finetune(input,lr, iterations);

    }

    /* delta computation for back prop with precon for SFH */
    public  List<Pair<INDArray,INDArray>>  computeDeltas2() {
        List<Pair<INDArray,INDArray>> deltaRet = new ArrayList<>();
        List<INDArray> activations = feedForward();
        INDArray[] deltas = new INDArray[activations.size() - 1];
        INDArray[] preCons = new INDArray[activations.size() - 1];

        //- y - h
        INDArray ix = activations.get(activations.size() - 1).sub(labels).divi(labels.rows());
       	/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();

        List<ActivationFunction> activationFunctions = new ArrayList<>();
        for(int j = 0; j < getNeuralNets().length; j++) {
            weights.add(getNeuralNets()[j].getW());
            biases.add(getNeuralNets()[j].gethBias());
            AutoEncoder a = (AutoEncoder) getNeuralNets()[j];
            activationFunctions.add(a.conf().getActivationFunction());
        }

        biases.add(getOutputLayer().getB());
        weights.add(getOutputLayer().getW());
        activationFunctions.add(getOutputLayer().conf().getActivationFunction());



        //errors
        for(int i = weights.size() - 1; i >= 0; i--) {
            deltas[i] = activations.get(i).transpose().mmul(ix);
            preCons[i] = Transforms.pow(activations.get(i).transpose(), 2).mmul(Transforms.pow(ix.dup(), 2)).muli(labels.rows());
            applyDropConnectIfNecessary(deltas[i]);

            if(i > 0) {
                //W[i] + b[i] * f'(z[i - 1])
                ix = ix.mmul(weights.get(i).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));
            }
        }

        for(int i = 0; i < deltas.length; i++) {
            if(layerWiseConfigurations.get(i).isConstrainGradientToUnitNorm())
                deltaRet.add(new Pair<>(deltas[i].divi(deltas[i].norm2(Integer.MAX_VALUE)),preCons[i]));

            else
                deltaRet.add(new Pair<>(deltas[i],preCons[i]));

        }

        return deltaRet;
    }






    public BaseMultiLayerNetwork getEncoder() {
        return encoder;
    }

    public void setEncoder(BaseMultiLayerNetwork encoder) {
        this.encoder = encoder;
    }





    public void setRoundCodeLayerInput(boolean roundCodeLayerInput) {
        this.roundCodeLayerInput = roundCodeLayerInput;
    }

    public ActivationFunction getOutputLayerActivation() {
        return outputLayerActivation;
    }

    public void setOutputLayerActivation(ActivationFunction outputLayerActivation) {
        this.outputLayerActivation = outputLayerActivation;
    }

    public boolean isNormalizeCodeLayerOutput() {
        return normalizeCodeLayerOutput;
    }

    public void setNormalizeCodeLayerOutput(boolean normalizeCodeLayerOutput) {
        this.normalizeCodeLayerOutput = normalizeCodeLayerOutput;
    }

    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public float score(DataSet data) {
        return 0;
    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        return 0;
    }

    /**
     * Returns the probabilities for each label
     * for each example row wise
     *
     * @param examples the examples to classify (one example in each row)
     * @return the likelihoods of each example and each label
     */
    @Override
    public INDArray labelProbabilities(INDArray examples) {
        return null;
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray examples, INDArray labels) {

    }

    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(DataSet data) {

    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {

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


    public static class Builder extends BaseMultiLayerNetwork.Builder<DeepAutoEncoder> {
        private  BaseMultiLayerNetwork encoder;

        public Builder() {
            clazz = DeepAutoEncoder.class;
        }

        public Builder withEncoder(BaseMultiLayerNetwork encoder) {
            this.encoder = encoder;
            return this;
        }




        /**
         * Use gauss newton back prop - this is for hessian free
         *
         * @param useGaussNewtonVectorProductBackProp whether to use gauss newton vector backprop
         * @return
         */
        @Override
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
        public Builder useDropConnection(boolean useDropConnect) {
            super.useDropConnection(useDropConnect);
            return this;
        }


        @Override
        public Builder lineSearchBackProp(boolean lineSearchBackProp) {
            super.lineSearchBackProp(lineSearchBackProp);
            return this;
        }



        @Override
        public Builder withVisibleBiasTransforms(Map<Integer, MatrixTransform> visibleBiasTransforms) {
            super.withVisibleBiasTransforms(visibleBiasTransforms);
            return this;
        }

        @Override
        public Builder withHiddenBiasTransforms(Map<Integer, MatrixTransform> hiddenBiasTransforms) {
            super.withHiddenBiasTransforms(hiddenBiasTransforms);
            return this;
        }

        /**
         * Forces use of number of epochs for training
         * SGD style rather than conjugate gradient
         *
         * @return
         */
        @Override
        public Builder forceEpochs() {
            super.forceEpochs();
            return this;
        }

        /**
         * Disables back propagation
         *
         * @return
         */
        @Override
        public Builder disableBackProp() {
            super.disableBackProp();
            return this;
        }

        /**
         * Transform the weights at the given layer
         *
         * @param layer     the layer to applyTransformToOrigin
         * @param transform the function used for transformation
         * @return
         */
        @Override
        public Builder transformWeightsAt(int layer, MatrixTransform transform) {
            super.transformWeightsAt(layer, transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given neuralNets
         *
         * @param transforms
         * @return
         */
        @Override
        public Builder transformWeightsAt(Map<Integer, MatrixTransform> transforms) {
            super.transformWeightsAt(transforms);
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
            super.withClazz(clazz);
            return this;
        }

        @Override
        public DeepAutoEncoder buildEmpty() {
            return  super.buildEmpty();

        }

        @Override
        public DeepAutoEncoder build() {


            //everything but output layer
            int inverseCount = encoder.getNeuralNets().length - 1;
            NeuralNetwork[] autoEncoders = new NeuralNetwork[encoder.getNeuralNets().length * 2 - 1];
            Layer[] hiddenLayers = new Layer[encoder.getNeuralNets().length * 2 - 1];
            for(int i = 0; i < autoEncoders.length; i++) {
                if(i < encoder.getNeuralNets().length) {
                    AutoEncoder a = new AutoEncoder.Builder().configure(encoder.getNeuralNets()[i].conf())
                            .withVisibleBias(encoder.getNeuralNets()[i].getvBias().dup())
                            .withHBias(encoder.getNeuralNets()[i].gethBias().dup())
                            .build();



                    Layer h = (Layer) encoder.getNeuralNets()[i].clone();
                    hiddenLayers[i] = h;
                    autoEncoders[i] = a;
                    if(i == encoder.getNeuralNets().length - 1)
                        a.conf().setActivationFunction(Activations.linear());

                }
                else {
                    AutoEncoder a = new AutoEncoder.Builder()
                            .configure(encoder.getNeuralNets()[inverseCount].conf())
                            .withWeights(encoder.getNeuralNets()[inverseCount].getW().transpose())
                            .withVisibleBias(encoder.getNeuralNets()[inverseCount].gethBias().dup())
                            .withHBias(encoder.getNeuralNets()[inverseCount].getvBias().dup())
                            .build();

                    autoEncoders[i] = a;
                    hiddenLayers[i] = (Layer) encoder.getNeuralNets()[inverseCount].transpose();
                    inverseCount--;
                }
            }

            OutputLayer o = new OutputLayer.Builder()
                    .withBias(encoder.getNeuralNets()[0].getvBias())
                    .withWeights(encoder.getNeuralNets()[0].getW().transpose())
                    .build();




            DeepAutoEncoder e = new DeepAutoEncoder();
            e.setLayers(autoEncoders);
            e.setUseDropConnect(encoder.isUseDropConnect());
            e.setUseGaussNewtonVectorProductBackProp(encoder.isUseGaussNewtonVectorProductBackProp());
            e.setSampleFromHiddenActivations(encoder.isSampleFromHiddenActivations());
            e.setLineSearchBackProp(encoder.isLineSearchBackProp());
            e.setForceNumEpochs(shouldForceEpochs);


            return e;

        }



    }


}
