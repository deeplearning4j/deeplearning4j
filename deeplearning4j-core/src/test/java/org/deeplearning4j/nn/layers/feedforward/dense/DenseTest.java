package org.deeplearning4j.nn.layers.feedforward.dense;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * Created by nyghtowl on 8/31/15.
 */
public class DenseTest extends BaseDL4JTest {

    private int numSamples = 150;
    private int batchSize = 150;
    private DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
    private DataSet data;

    @Test
    public void testDenseBiasInit() {
        DenseLayer build = new DenseLayer.Builder().nIn(1).nOut(3).biasInit(1).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layer(build).build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);

        assertEquals(1, layer.getParam("b").size(0));
    }

    @Test
    public void testMLPMultiLayerPretrain() {
        // Note CNN does not do pretrain
        MultiLayerNetwork model = getDenseMLNConfig(false, true);
        model.fit(iter);

        MultiLayerNetwork model2 = getDenseMLNConfig(false, true);
        model2.fit(iter);
        iter.reset();

        DataSet test = iter.next();

        assertEquals(model.params(), model2.params());

        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        double f1Score = eval.f1();

        Evaluation eval2 = new Evaluation();
        INDArray output2 = model2.output(test.getFeatureMatrix());
        eval2.eval(test.getLabels(), output2);
        double f1Score2 = eval2.f1();

        assertEquals(f1Score, f1Score2, 1e-4);

    }

    @Test
    public void testMLPMultiLayerBackprop() {
        MultiLayerNetwork model = getDenseMLNConfig(true, false);
        model.fit(iter);

        MultiLayerNetwork model2 = getDenseMLNConfig(true, false);
        model2.fit(iter);
        iter.reset();

        DataSet test = iter.next();

        assertEquals(model.params(), model2.params());

        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        double f1Score = eval.f1();

        Evaluation eval2 = new Evaluation();
        INDArray output2 = model2.output(test.getFeatureMatrix());
        eval2.eval(test.getLabels(), output2);
        double f1Score2 = eval2.f1();

        assertEquals(f1Score, f1Score2, 1e-4);

    }


    //////////////////////////////////////////////////////////////////////////////////

    private static MultiLayerNetwork getDenseMLNConfig(boolean backprop, boolean pretrain) {
        int numInputs = 4;
        int outputNum = 3;
        long seed = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
                        .updater(new Sgd(1e-3)).l1(0.3).l2(1e-3).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(numInputs).nOut(3)
                                        .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(3).nOut(2)
                                        .activation(Activation.TANH).weightInit(WeightInit.XAVIER).build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .weightInit(WeightInit.XAVIER).nIn(2).nOut(outputNum).build())
                        .backprop(backprop).pretrain(pretrain).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;

    }
}
