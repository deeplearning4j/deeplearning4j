package org.deeplearning4j.models.layers;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.ConvolutionDownSampleLayer;
import org.junit.Test;
import org.junit.Ignore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * Created by agibsonccc on 9/7/14.
 */
public class ConvolutionDownSampleLayerTest {


    @Test
    @Ignore
    public void testConvolution() throws Exception {
        MnistDataFetcher data = new MnistDataFetcher(true);
        data.fetch(2);
        DataSet d = data.next();

        d.setFeatures(d.getFeatureMatrix().reshape(2,1,28,28));

        NeuralNetConfiguration n = new NeuralNetConfiguration.Builder()
                .filterSize(new int[]{2,2}).numFeatureMaps(2)
                .weightShape(new int[]{2, 3, 9, 9}).build();

        ConvolutionDownSampleLayer c = new ConvolutionDownSampleLayer(n);

        INDArray convolved = c.activate(d.getFeatureMatrix());


    }


}
