package org.deeplearning4j.models.layers;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.ConvolutionLayer;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Created by agibsonccc on 9/7/14.
 */
public class ConvolutionLayerTest {


    @Test
    public void testConvolution() throws Exception {
        MnistDataFetcher data = new MnistDataFetcher(true);
        data.fetch(2);
        DataSet d = data.next();

        d.setFeatures(d.getFeatureMatrix().reshape(2,1,28,28));

        NeuralNetConfiguration n = new NeuralNetConfiguration.Builder()
                .filterSize(new int[]{2,2}).numFeatureMaps(2)
                .weightShape(new int[]{2,3,9,9}).build();

        ConvolutionLayer c = new ConvolutionLayer(n);

        c.activate(d.getFeatureMatrix());

    }


}
