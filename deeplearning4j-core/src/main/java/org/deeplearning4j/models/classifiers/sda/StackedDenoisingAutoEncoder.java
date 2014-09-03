package org.deeplearning4j.models.classifiers.sda;

import org.deeplearning4j.models.featuredetectors.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.NeuralNetwork;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


/**
 *  Stacked Denoising AutoEncoders are merely denoising autoencoders
 *  who's inputs feed in to the next one.
 * @author Adam Gibson
 *
 */
public class StackedDenoisingAutoEncoder extends BaseMultiLayerNetwork {

    private static final long serialVersionUID = 1448581794985193009L;
    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoder.class);


    public void pretrain(float lr, float corruptionLevel, int epochs) {
        pretrain(this.getInput(), lr, corruptionLevel, epochs);
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
        float corruptionLevel = (float) otherParams[0];
        float lr = (Float) otherParams[1];
        int epochs = (Integer) otherParams[2];
        int passes = otherParams.length > 3 ? (Integer) otherParams[3] : 1;
        for (int i = 0; i < passes; i++)
            pretrain(iter, corruptionLevel, lr, epochs);

    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     *
     * @param iter            the input to iterate on
     * @param corruptionLevel the corruption level to use for running the denoising autoencoder training.
     *                        The typical tip is that the higher k is the closer to the model
     *                        you will be approximating due to more sampling. K = 1
     *                        usually gives very good results and is the default in quite a few situations.
     * @param lr              the learning rate to use
     * @param iterations      the number of epochs to iterate
     */
    public void pretrain(DataSetIterator iter, float corruptionLevel, float lr, int iterations) {

        INDArray layerInput;

        for (int i = 0; i < getnLayers(); i++) {
            if (i == 0) {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    this.input = next.getFeatureMatrix();
                      /*During pretrain, feed forward expected activations of network, use activation functions during pretrain  */
                    if (this.getInput() == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null || this.getNeuralNets() == null || this.getNeuralNets()[0] == null) {
                        setInput(input);
                        initializeLayers(input);
                    } else
                        setInput(input);
                    //override learning rate where present
                    float realLearningRate = layerWiseConfigurations.get(i).getLr();
                    if (isForceNumEpochs()) {
                        for (int iteration = 0; iteration < iterations; iteration++) {
                            log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                            getNeuralNets()[i].iterate(next.getFeatureMatrix(), new Object[]{corruptionLevel, lr});
                            getNeuralNets()[i].iterationDone(iteration);
                        }
                    } else
                        getNeuralNets()[i].fit(next.getFeatureMatrix(), new Object[]{corruptionLevel, realLearningRate, iterations});

                }

                iter.reset();
            } else {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    layerInput = next.getFeatureMatrix();
                    for (int j = 1; j <= i; j++)
                           layerInput = activationFromPrevLayer(j,layerInput);



                    log.info("Training on layer " + (i + 1));
                    //override learning rate where present
                    float realLearningRate = layerWiseConfigurations.get(i).getLr();
                    if (isForceNumEpochs()) {
                        for (int iteration = 0; iteration < iterations; iteration++) {
                            log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                            getNeuralNets()[i].iterate(layerInput, new Object[]{corruptionLevel, lr});
                            getNeuralNets()[i].iterationDone(iteration);
                        }
                    } else
                        getNeuralNets()[i].fit(layerInput, new Object[]{corruptionLevel, realLearningRate, iterations});


                }

                iter.reset();

            }
        }

    }

    @Override
    public void pretrain(INDArray input, Object[] otherParams) {
        if (otherParams == null) {
            otherParams = new Object[]{0.01, 0.3, 1000};
        }

        Float lr = (Float) otherParams[0];
        Float corruptionLevel = (Float) otherParams[1];
        Integer iterations = (Integer) otherParams[2];

        pretrain(input, lr, corruptionLevel, iterations);

    }

    /**
     * Unsupervised pretraining based on reconstructing the input
     * from a corrupted version
     *
     * @param input           the input to iterate on
     * @param lr              the starting learning rate
     * @param corruptionLevel the corruption level (the smaller number of inputs; the higher the
     *                        corruption level should be) the percent of inputs to corrupt
     * @param iterations      the number of iterations to run
     */
    public void pretrain(INDArray input, float lr, float corruptionLevel, int iterations) {
        if (this.getInput() == null)
            initializeLayers(input.dup());

        if (isUseGaussNewtonVectorProductBackProp())
            log.warn("Warning; using gauss newton vector back prop with pretrain is known to cause issues with obscenely large activations.");

        this.input = input;

        INDArray layerInput = null;

        for (int i = 0; i < this.getnLayers(); i++) {  // layer-wise
            //input layer
            if (i == 0)
                layerInput = input;
            else
                layerInput = this.getNeuralNets()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();
            if (isForceNumEpochs()) {
                for (int iteration = 0; iteration < iterations; iteration++) {
                    getNeuralNets()[i].iterate(layerInput, new Object[]{corruptionLevel, lr});
                    log.info("Error on iteration " + iteration + " for layer " + (i + 1) + " is " + getNeuralNets()[i].score());
                    getNeuralNets()[i].iterationDone(iteration);

                }
            } else
                getNeuralNets()[i].fit(layerInput, new Object[]{corruptionLevel, lr, iterations});


        }
    }


    @Override
    public NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hbias,
                                     INDArray vBias, int index) {
        DenoisingAutoEncoder ret = new DenoisingAutoEncoder.Builder().configure(layerWiseConfigurations.get(index))
                .withInput(input).withWeights(W).withHBias(hbias).withVisibleBias(vBias)
                .build();
        return ret;
    }


    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new DenoisingAutoEncoder[numLayers];
    }

    @Override
    public Layer createHiddenLayer(int index, int nIn, int nOut, INDArray layerInput) {
        return null;
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


    public static class Builder extends BaseMultiLayerNetwork.Builder<StackedDenoisingAutoEncoder> {
        public Builder() {
            this.clazz = StackedDenoisingAutoEncoder.class;
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
        public Builder useDropConnection(boolean useDropConnect) {
            super.useDropConnection(useDropConnect);
            return this;
        }


        @Override
        public Builder lineSearchBackProp(boolean lineSearchBackProp) {
            super.lineSearchBackProp(lineSearchBackProp);
            return this;
        }


        public Builder withVisibleBiasTransforms(Map<Integer, MatrixTransform> visibleBiasTransforms) {
            super.withVisibleBiasTransforms(visibleBiasTransforms);
            return this;
        }

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
        public Builder forceEpochs() {
            shouldForceEpochs = true;
            return this;
        }

        /**
         * Disables back propagation
         *
         * @return
         */
        public Builder disableBackProp() {
            backProp = false;
            return this;
        }

        /**
         * Transform the weights at the given layer
         *
         * @param layer     the layer to applyTransformToOrigin
         * @param transform the function used for transformation
         * @return
         */
        public Builder transformWeightsAt(int layer, MatrixTransform transform) {
            weightTransforms.put(layer, transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given neuralNets
         *
         * @param transforms
         * @return
         */
        public Builder transformWeightsAt(Map<Integer, MatrixTransform> transforms) {
            weightTransforms.putAll(transforms);
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



        public Builder withInput(INDArray input) {
            super.withInput(input);
            return this;
        }


        public Builder withLabels(INDArray labels) {
            super.withLabels(labels);
            return this;
        }

        public Builder withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz = clazz;
            return this;
        }

    }
}







