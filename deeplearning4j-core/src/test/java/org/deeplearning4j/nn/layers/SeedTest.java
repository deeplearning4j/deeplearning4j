package org.deeplearning4j.nn.layers;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import static org.junit.Assert.assertEquals;

/**
 */

public class SeedTest {

    private DataSetIterator irisIter = new IrisDataSetIterator(50,50);
    private DataSet data = irisIter.next();


    @Test
    public void testAutoEncoderSeed() {
        AutoEncoder layerType = new AutoEncoder.Builder()
                .nIn(4)
                .nOut(3).corruptionLevel(0.0)
                .activation("sigmoid")
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .layer(layerType)
                .seed(123)
                .build();

        Layer layer = LayerFactories.getFactory(conf).create(conf);
        layer.fit(data.getFeatureMatrix());

        double score = layer.score();
        INDArray parameters = layer.params();
        layer.setParams(parameters);
        layer.computeGradientAndScore();

        double score2 = layer.score();
        assertEquals(parameters, layer.params());
        assertEquals(score, score2, 1e-4);
    }


    @Test
    public void testRecursiveAutoEncoderSeed() {
        RecursiveAutoEncoder layerType = new RecursiveAutoEncoder.Builder()
                .nIn(4)
                .nOut(3)
                .activation("sigmoid")
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .layer(layerType)
                .seed(123)
                .build();

        Layer layer = LayerFactories.getFactory(conf).create(conf);
        layer.fit(data.getFeatureMatrix());

        double score = layer.score();
        INDArray parameters = layer.params();
        layer.setParams(parameters);
        layer.computeGradientAndScore();

        double score2 = layer.score();
        assertEquals(parameters, layer.params());
        assertEquals(score, score2, 1e-1);
    }


}
