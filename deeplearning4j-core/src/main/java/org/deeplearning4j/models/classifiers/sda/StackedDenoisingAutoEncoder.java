package org.deeplearning4j.models.classifiers.sda;

import org.deeplearning4j.models.featuredetectors.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.transformation.MatrixTransform;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
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

                    getNeuralNets()[i].fit(next.getFeatureMatrix());

                }

                iter.reset();
            } else {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    layerInput = next.getFeatureMatrix();
                    for (int j = 1; j <= i; j++)
                        layerInput = activationFromPrevLayer(j,layerInput);



                    log.info("Training on layer " + (i + 1));
                    getNeuralNets()[i].fit(layerInput);


                }

                iter.reset();

            }
        }

    }


    @Override
    public void pretrain(DataSetIterator iter) {
        if(!layerWiseConfigurations.isPretrain())
            return;

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
                    getNeuralNets()[i].fit(next.getFeatureMatrix());

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
                    getNeuralNets()[i].fit(layerInput);

                }

                iter.reset();


            }
        }
    }

    /**
     * Unsupervised pretraining based on reconstructing the input
     * from a corrupted version
     *
     * @param input           the input to iterate on
     */
    public void pretrain(INDArray input) {

        if (this.getInput() == null)
            initializeLayers(input.dup());

        if(!layerWiseConfigurations.isPretrain())
            return;

        this.input = input;

        INDArray layerInput = null;

        for (int i = 0; i < this.getnLayers(); i++) {  // layer-wise
            //input layer
            if (i == 0)
                layerInput = input;
            else
                layerInput = this.getNeuralNets()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();
            getNeuralNets()[i].fit(layerInput);


        }
    }



    @Override
    public NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hbias,
                                     INDArray vBias, int index) {
        DenoisingAutoEncoder ret = new DenoisingAutoEncoder.Builder().configure(layerWiseConfigurations.getConf(index))
                .withInput(input).withWeights(W).withHBias(hbias).withVisibleBias(vBias)
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

        @Override
        public Builder configure(NeuralNetConfiguration conf) {
            super.configure(conf);
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

        @Override
        public Builder layerWiseConfiguration(MultiLayerConfiguration layerWiseConfiguration) {
            super.layerWiseConfiguration(layerWiseConfiguration);
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

        public StackedDenoisingAutoEncoder build() {
            StackedDenoisingAutoEncoder ret = super.build();
            if(ret.defaultConfiguration == null)
                ret.defaultConfiguration = this.multiLayerConfiguration.getConf(0);
            ret.initializeLayers(Nd4j.zeros(1, ret.defaultConfiguration.getnIn()));

            return ret;
        }


    }
}







