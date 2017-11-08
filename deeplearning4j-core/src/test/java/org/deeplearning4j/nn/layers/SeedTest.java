package org.deeplearning4j.nn.layers;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.conf.GlobalConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 */

public class SeedTest {

    private DataSetIterator irisIter = new IrisDataSetIterator(50, 50);
    private DataSet data = irisIter.next();


    @Test
    public void testAutoEncoderSeed() {
        AutoEncoder l = new AutoEncoder.Builder().nIn(4).nOut(3).corruptionLevel(0.0)
                        .activation(Activation.SIGMOID).build();
        l.applyGlobalConfiguration(new GlobalConfiguration());

        int numParams = l.initializer().numParams(l);
        INDArray params = Nd4j.create(1, numParams);
        Model layer = (Model)l.instantiate(null, null, 0, 1, params, true);
        layer.setBackpropGradientsViewArray(Nd4j.create(1, numParams));
        Activations a = ActivationsFactory.getInstance().create(data.getFeatures());
        layer.fit(a);

        layer.computeGradientAndScore(a, null);
        double score = layer.score();
        INDArray parameters = layer.params();
        layer.setParams(parameters);
        layer.computeGradientAndScore(a, null);

        double score2 = layer.score();
        assertEquals(parameters, layer.params());
        assertEquals(score, score2, 1e-4);
    }
}
