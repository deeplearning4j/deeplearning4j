package org.deeplearning4j.models.classifiers.sae;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.featuredetectors.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by mjk on 9/17/14.
 */
public class StackedAutoEncoder extends BaseMultiLayerNetwork {
    @Override
    public NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hbias, INDArray vBias, int index) {
        AutoEncoder ret = new AutoEncoder.Builder().configure(layerWiseConfigurations.get(index))
                .withInput(input).withWeights(W).withHBias(hbias).withVisibleBias(vBias)
                .build();
        return ret;
    }

    @Override
    public void pretrain(DataSetIterator iter, Object[] otherParams) {

    }

    @Override
    public void pretrain(INDArray input, Object[] otherParams) {

    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new AutoEncoder[numLayers];
    }

    @Override
    public INDArray transform(INDArray data) {
        return null;
    }


    @Override
    public void fit(INDArray data, Object[] params) {
        //fit is already defined in the basemultilayernetwork
    }
}
