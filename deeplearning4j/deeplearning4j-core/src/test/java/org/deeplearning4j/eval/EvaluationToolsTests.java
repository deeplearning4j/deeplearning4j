package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by Alex on 07/01/2017.
 */
public class EvaluationToolsTests extends BaseDL4JTest {

    @Test
    public void testRocHtml() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH).build()).layer(1,
                                        new OutputLayer.Builder().nIn(4).nOut(2).activation(Activation.SOFTMAX)
                                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        NormalizerStandardize ns = new NormalizerStandardize();
        DataSet ds = iter.next();
        ns.fit(ds);
        ns.transform(ds);

        INDArray newLabels = Nd4j.create(150, 2);
        newLabels.getColumn(0).assign(ds.getLabels().getColumn(0));
        newLabels.getColumn(0).addi(ds.getLabels().getColumn(1));
        newLabels.getColumn(1).assign(ds.getLabels().getColumn(2));
        ds.setLabels(newLabels);

        for (int i = 0; i < 30; i++) {
            net.fit(ds);
        }

        for (int numSteps : new int[] {20, 0}) {
            ROC roc = new ROC(numSteps);
            iter.reset();

            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            INDArray out = net.output(f);
            roc.eval(l, out);


            String str = EvaluationTools.rocChartToHtml(roc);
            //            System.out.println(str);
        }
    }

    @Test
    public void testRocMultiToHtml() throws Exception {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(4).activation(Activation.TANH).build()).layer(1,
                                        new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        NormalizerStandardize ns = new NormalizerStandardize();
        DataSet ds = iter.next();
        ns.fit(ds);
        ns.transform(ds);

        for (int i = 0; i < 30; i++) {
            net.fit(ds);
        }

        for (int numSteps : new int[] {20, 0}) {
            ROCMultiClass roc = new ROCMultiClass(numSteps);
            iter.reset();

            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            INDArray out = net.output(f);
            roc.eval(l, out);


            String str = EvaluationTools.rocChartToHtml(roc, Arrays.asList("setosa", "versicolor", "virginica"));
            System.out.println(str);
        }
    }

    @Test
    public void testEvaluationCalibrationToHtml() throws Exception {
        int minibatch = 1000;
        int nClasses = 3;

        INDArray arr = Nd4j.rand(minibatch, nClasses);
        arr.diviColumnVector(arr.sum(1));
        INDArray labels = Nd4j.zeros(minibatch, nClasses);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nClasses), 1.0);
        }

        int numBins = 10;
        EvaluationCalibration ec = new EvaluationCalibration(numBins, numBins);
        ec.eval(labels, arr);

        String str = EvaluationTools.evaluationCalibrationToHtml(ec);
        //        System.out.println(str);
    }

}
