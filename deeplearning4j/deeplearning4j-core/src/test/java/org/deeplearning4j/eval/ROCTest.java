package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.curves.PrecisionRecallCurve;
import org.deeplearning4j.eval.curves.RocCurve;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by Alex on 04/11/2016.
 */
public class ROCTest extends BaseDL4JTest {

    private static Map<Double, Double> expTPR;
    private static Map<Double, Double> expFPR;

    static {
        expTPR = new HashMap<>();
        double totalPositives = 5.0;
        expTPR.put(0 / 10.0, 5.0 / totalPositives); //All 10 predicted as class 1, of which 5 of 5 are correct
        expTPR.put(1 / 10.0, 5.0 / totalPositives);
        expTPR.put(2 / 10.0, 5.0 / totalPositives);
        expTPR.put(3 / 10.0, 5.0 / totalPositives);
        expTPR.put(4 / 10.0, 5.0 / totalPositives);
        expTPR.put(5 / 10.0, 5.0 / totalPositives);
        expTPR.put(6 / 10.0, 4.0 / totalPositives); //Threshold: 0.4 -> last 4 predicted; last 5 actual
        expTPR.put(7 / 10.0, 3.0 / totalPositives);
        expTPR.put(8 / 10.0, 2.0 / totalPositives);
        expTPR.put(9 / 10.0, 1.0 / totalPositives);
        expTPR.put(10 / 10.0, 0.0 / totalPositives);

        expFPR = new HashMap<>();
        double totalNegatives = 5.0;
        expFPR.put(0 / 10.0, 5.0 / totalNegatives); //All 10 predicted as class 1, but all 5 true negatives are predicted positive
        expFPR.put(1 / 10.0, 4.0 / totalNegatives); //1 true negative is predicted as negative; 4 false positives
        expFPR.put(2 / 10.0, 3.0 / totalNegatives); //2 true negatives are predicted as negative; 3 false positives
        expFPR.put(3 / 10.0, 2.0 / totalNegatives);
        expFPR.put(4 / 10.0, 1.0 / totalNegatives);
        expFPR.put(5 / 10.0, 0.0 / totalNegatives);
        expFPR.put(6 / 10.0, 0.0 / totalNegatives);
        expFPR.put(7 / 10.0, 0.0 / totalNegatives);
        expFPR.put(8 / 10.0, 0.0 / totalNegatives);
        expFPR.put(9 / 10.0, 0.0 / totalNegatives);
        expFPR.put(10 / 10.0, 0.0 / totalNegatives);
    }

    @Test
    public void testRocBasic() {
        //2 outputs here - probability distribution over classes (softmax)
        INDArray predictions = Nd4j.create(new double[][] {{1.0, 0.001}, //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
                        {0.899, 0.101}, {0.799, 0.201}, {0.699, 0.301}, {0.599, 0.401}, {0.499, 0.501}, {0.399, 0.601},
                        {0.299, 0.701}, {0.199, 0.801}, {0.099, 0.901}});

        INDArray actual = Nd4j.create(new double[][] {{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1},
                        {0, 1}, {0, 1}});

        ROC roc = new ROC(10);
        roc.eval(actual, predictions);

        RocCurve rocCurve = roc.getRocCurve();

        assertEquals(11, rocCurve.getThreshold().length); //0 + 10 steps
        for (int i = 0; i < 11; i++) {
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, rocCurve.getThreshold(i), 1e-5);

            //            System.out.println("t=" + expThreshold + "\t" + v.getFalsePositiveRate() + "\t" + v.getTruePositiveRate());

            double efpr = expFPR.get(expThreshold);
            double afpr = rocCurve.getFalsePositiveRate(i);
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = rocCurve.getTruePositiveRate(i);
            assertEquals(etpr, atpr, 1e-5);
        }


        //Expect AUC == 1.0 here
        double auc = roc.calculateAUC();
        assertEquals(1.0, auc, 1e-6);

        // testing reset now
        roc.reset();
        roc.eval(actual, predictions);
        auc = roc.calculateAUC();
        assertEquals(1.0, auc, 1e-6);
    }

    @Test
    public void testRocBasicSingleClass() {
        //1 output here - single probability value (sigmoid)

        //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
        INDArray predictions =
                        Nd4j.create(new double[] {0.001, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901},
                                        new int[] {10, 1});

        INDArray actual = Nd4j.create(new double[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, new int[] {10, 1});

        ROC roc = new ROC(10);
        roc.eval(actual, predictions);

        RocCurve rocCurve = roc.getRocCurve();

        assertEquals(11, rocCurve.getThreshold().length); //0 + 10 steps
        for (int i = 0; i < 11; i++) {
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, rocCurve.getThreshold(i), 1e-5);

            //            System.out.println("t=" + expThreshold + "\t" + v.getFalsePositiveRate() + "\t" + v.getTruePositiveRate());

            double efpr = expFPR.get(expThreshold);
            double afpr = rocCurve.getFalsePositiveRate(i);
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = rocCurve.getTruePositiveRate(i);
            assertEquals(etpr, atpr, 1e-5);
        }

        //Expect AUC == 1.0 here
        double auc = roc.calculateAUC();
        assertEquals(1.0, auc, 1e-6);
    }


    @Test
    public void testRoc() {
        //Previous tests allowed for a perfect classifier with right threshold...

        INDArray labels = Nd4j.create(new double[][] {{0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}});

        INDArray prediction = Nd4j.create(new double[][] {{0.199, 0.801}, {0.499, 0.501}, {0.399, 0.601},
                        {0.799, 0.201}, {0.899, 0.101}});

        Map<Double, Double> expTPR = new HashMap<>();
        double totalPositives = 2.0;
        expTPR.put(0.0, 2.0 / totalPositives); //All predicted class 1 -> 2 true positives / 2 total positives
        expTPR.put(0.1, 2.0 / totalPositives);
        expTPR.put(0.2, 2.0 / totalPositives);
        expTPR.put(0.3, 2.0 / totalPositives);
        expTPR.put(0.4, 2.0 / totalPositives);
        expTPR.put(0.5, 2.0 / totalPositives);
        expTPR.put(0.6, 1.0 / totalPositives); //At threshold of 0.6, only 1 of 2 positives are predicted positive
        expTPR.put(0.7, 1.0 / totalPositives);
        expTPR.put(0.8, 1.0 / totalPositives);
        expTPR.put(0.9, 0.0 / totalPositives); //At threshold of 0.9, 0 of 2 positives are predicted positive
        expTPR.put(1.0, 0.0 / totalPositives);

        Map<Double, Double> expFPR = new HashMap<>();
        double totalNegatives = 3.0;
        expFPR.put(0.0, 3.0 / totalNegatives); //All predicted class 1 -> 3 false positives / 3 total negatives
        expFPR.put(0.1, 3.0 / totalNegatives);
        expFPR.put(0.2, 2.0 / totalNegatives); //At threshold of 0.2: 1 true negative, 2 false positives
        expFPR.put(0.3, 1.0 / totalNegatives); //At threshold of 0.3: 2 true negative, 1 false positive
        expFPR.put(0.4, 1.0 / totalNegatives);
        expFPR.put(0.5, 1.0 / totalNegatives);
        expFPR.put(0.6, 1.0 / totalNegatives);
        expFPR.put(0.7, 0.0 / totalNegatives); //At threshold of 0.7: 3 true negatives, 0 false positives
        expFPR.put(0.8, 0.0 / totalNegatives);
        expFPR.put(0.9, 0.0 / totalNegatives);
        expFPR.put(1.0, 0.0 / totalNegatives);

        int[] expTPs = new int[] {2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0};
        int[] expFPs = new int[] {3, 3, 2, 1, 1, 1, 1, 0, 0, 0, 0};
        int[] expFNs = new int[11];
        int[] expTNs = new int[11];
        for (int i = 0; i < 11; i++) {
            expFNs[i] = (int) totalPositives - expTPs[i];
            expTNs[i] = 5 - expTPs[i] - expFPs[i] - expFNs[i];
        }

        ROC roc = new ROC(10);
        roc.eval(labels, prediction);

        RocCurve rocCurve = roc.getRocCurve();

        assertEquals(11, rocCurve.getThreshold().length);
        assertEquals(11, rocCurve.getFpr().length);
        assertEquals(11, rocCurve.getTpr().length);

        for (int i = 0; i < 11; i++) {
            double expThreshold = i / 10.0;
            assertEquals(expThreshold, rocCurve.getThreshold(i), 1e-5);

            double efpr = expFPR.get(expThreshold);
            double afpr = rocCurve.getFalsePositiveRate(i);
            assertEquals(efpr, afpr, 1e-5);

            double etpr = expTPR.get(expThreshold);
            double atpr = rocCurve.getTruePositiveRate(i);
            assertEquals(etpr, atpr, 1e-5);
        }

        //AUC: expected values are based on plotting the ROC curve and manually calculating the area
        double expAUC = 0.5 * 1.0 / 3.0 + (1 - 1 / 3.0) * 1.0;
        double actAUC = roc.calculateAUC();

        assertEquals(expAUC, actAUC, 1e-6);

        PrecisionRecallCurve prc = roc.getPrecisionRecallCurve();
        for (int i = 0; i < 11; i++) {
            PrecisionRecallCurve.Confusion c = prc.getConfusionMatrixAtThreshold(i * 0.1);
            assertEquals(expTPs[i], c.getTpCount());
            assertEquals(expFPs[i], c.getFpCount());
            assertEquals(expFPs[i], c.getFpCount());
            assertEquals(expTNs[i], c.getTnCount());
        }
    }


    @Test
    public void testRocTimeSeriesNoMasking() {
        //Same as first test...

        //2 outputs here - probability distribution over classes (softmax)
        INDArray predictions2d = Nd4j.create(new double[][] {{1.0, 0.001}, //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
                        {0.899, 0.101}, {0.799, 0.201}, {0.699, 0.301}, {0.599, 0.401}, {0.499, 0.501}, {0.399, 0.601},
                        {0.299, 0.701}, {0.199, 0.801}, {0.099, 0.901}});

        INDArray actual2d = Nd4j.create(new double[][] {{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1},
                        {0, 1}, {0, 1}});

        INDArray predictions3d = Nd4j.create(2, 2, 5);
        INDArray firstTSp =
                        predictions3d.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).transpose();
        assertArrayEquals(new long[] {5, 2}, firstTSp.shape());
        firstTSp.assign(predictions2d.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        INDArray secondTSp =
                        predictions3d.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).transpose();
        assertArrayEquals(new long[] {5, 2}, secondTSp.shape());
        secondTSp.assign(predictions2d.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));

        INDArray labels3d = Nd4j.create(2, 2, 5);
        INDArray firstTS = labels3d.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()).transpose();
        assertArrayEquals(new long[] {5, 2}, firstTS.shape());
        firstTS.assign(actual2d.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        INDArray secondTS = labels3d.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all()).transpose();
        assertArrayEquals(new long[] {5, 2}, secondTS.shape());
        secondTS.assign(actual2d.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));

        for (int steps : new int[] {10, 0}) { //0 steps: exact
            //            System.out.println("Steps: " + steps);
            ROC rocExp = new ROC(steps);
            rocExp.eval(actual2d, predictions2d);

            ROC rocAct = new ROC(steps);
            rocAct.evalTimeSeries(labels3d, predictions3d);

            assertEquals(rocExp.calculateAUC(), rocAct.calculateAUC(), 1e-6);
            assertEquals(rocExp.calculateAUCPR(), rocAct.calculateAUCPR(), 1e-6);

            assertEquals(rocExp.getRocCurve(), rocAct.getRocCurve());
        }
    }

    @Test
    public void testRocTimeSeriesMasking() {
        //2 outputs here - probability distribution over classes (softmax)
        INDArray predictions2d = Nd4j.create(new double[][] {{1.0, 0.001}, //add 0.001 to avoid numerical/rounding issues (float vs. double, etc)
                        {0.899, 0.101}, {0.799, 0.201}, {0.699, 0.301}, {0.599, 0.401}, {0.499, 0.501}, {0.399, 0.601},
                        {0.299, 0.701}, {0.199, 0.801}, {0.099, 0.901}});

        INDArray actual2d = Nd4j.create(new double[][] {{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {0, 1}, {0, 1}, {0, 1},
                        {0, 1}, {0, 1}});


        //Create time series data... first time series: length 4. Second time series: length 6
        INDArray predictions3d = Nd4j.create(2, 2, 6);
        INDArray tad = predictions3d.tensorAlongDimension(0, 1, 2).transpose();
        tad.get(NDArrayIndex.interval(0, 4), NDArrayIndex.all())
                        .assign(predictions2d.get(NDArrayIndex.interval(0, 4), NDArrayIndex.all()));

        tad = predictions3d.tensorAlongDimension(1, 1, 2).transpose();
        tad.assign(predictions2d.get(NDArrayIndex.interval(4, 10), NDArrayIndex.all()));


        INDArray labels3d = Nd4j.create(2, 2, 6);
        tad = labels3d.tensorAlongDimension(0, 1, 2).transpose();
        tad.get(NDArrayIndex.interval(0, 4), NDArrayIndex.all())
                        .assign(actual2d.get(NDArrayIndex.interval(0, 4), NDArrayIndex.all()));

        tad = labels3d.tensorAlongDimension(1, 1, 2).transpose();
        tad.assign(actual2d.get(NDArrayIndex.interval(4, 10), NDArrayIndex.all()));


        INDArray mask = Nd4j.zeros(2, 6);
        mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 4)).assign(1);
        mask.get(NDArrayIndex.point(1), NDArrayIndex.all()).assign(1);


        for (int steps : new int[] {20, 0}) { //0 steps: exact
            ROC rocExp = new ROC(steps);
            rocExp.eval(actual2d, predictions2d);

            ROC rocAct = new ROC(steps);
            rocAct.evalTimeSeries(labels3d, predictions3d, mask);

            assertEquals(rocExp.calculateAUC(), rocAct.calculateAUC(), 1e-6);

            assertEquals(rocExp.getRocCurve(), rocAct.getRocCurve());
        }
    }



    @Test
    public void testCompareRocAndRocMultiClass() {
        Nd4j.getRandom().setSeed(12345);

        //For 2 class case: ROC and Multi-class ROC should be the same...
        int nExamples = 200;
        INDArray predictions = Nd4j.rand(nExamples, 2);
        INDArray tempSum = predictions.sum(1);
        predictions.diviColumnVector(tempSum);

        INDArray labels = Nd4j.create(nExamples, 2);
        Random r = new Random(12345);
        for (int i = 0; i < nExamples; i++) {
            labels.putScalar(i, r.nextInt(2), 1.0);
        }

        for (int numSteps : new int[] {30, 0}) { //Steps = 0: exact
            ROC roc = new ROC(numSteps);
            roc.eval(labels, predictions);

            ROCMultiClass rocMultiClass = new ROCMultiClass(numSteps);
            rocMultiClass.eval(labels, predictions);

            double auc = roc.calculateAUC();
            double auc1 = rocMultiClass.calculateAUC(1);

            assertEquals(auc, auc1, 1e-6);
        }
    }

    @Test
    public void testCompare2Vs3Classes() {

        //ROC multi-class: 2 vs. 3 classes should be the same, if we add two of the classes together...
        //Both methods implement one vs. all ROC/AUC in different ways

        int nExamples = 200;
        INDArray predictions3 = Nd4j.rand(nExamples, 3);
        INDArray tempSum = predictions3.sum(1);
        predictions3.diviColumnVector(tempSum);

        INDArray labels3 = Nd4j.create(nExamples, 3);
        Random r = new Random(12345);
        for (int i = 0; i < nExamples; i++) {
            labels3.putScalar(i, r.nextInt(3), 1.0);
        }

        INDArray predictions2 = Nd4j.zeros(nExamples, 2);
        predictions2.getColumn(0).assign(predictions3.getColumn(0));
        predictions2.getColumn(0).addi(predictions3.getColumn(1));
        predictions2.getColumn(1).addi(predictions3.getColumn(2));

        INDArray labels2 = Nd4j.zeros(nExamples, 2);
        labels2.getColumn(0).assign(labels3.getColumn(0));
        labels2.getColumn(0).addi(labels3.getColumn(1));
        labels2.getColumn(1).addi(labels3.getColumn(2));

        for (int numSteps : new int[] {30, 0}) { //Steps = 0: exact

            ROCMultiClass rocMultiClass3 = new ROCMultiClass(numSteps);
            ROCMultiClass rocMultiClass2 = new ROCMultiClass(numSteps);

            rocMultiClass3.eval(labels3, predictions3);
            rocMultiClass2.eval(labels2, predictions2);

            double auc3 = rocMultiClass3.calculateAUC(2);
            double auc2 = rocMultiClass2.calculateAUC(1);

            assertEquals(auc2, auc3, 1e-6);

            RocCurve c3 = rocMultiClass3.getRocCurve(2);
            RocCurve c2 = rocMultiClass2.getRocCurve(1);

            assertArrayEquals(c2.getThreshold(), c3.getThreshold(), 1e-6);
            assertArrayEquals(c2.getFpr(), c3.getFpr(), 1e-6);
            assertArrayEquals(c2.getTpr(), c3.getTpr(), 1e-6);
        }
    }

    @Test
    public void testROCMerging() {
        int nArrays = 10;
        int minibatch = 64;
        int nROCs = 3;

        for (int steps : new int[] {0, 20}) { //0 steps: exact, 20 steps: thresholded

            Nd4j.getRandom().setSeed(12345);
            Random r = new Random(12345);

            List<ROC> rocList = new ArrayList<>();
            for (int i = 0; i < nROCs; i++) {
                rocList.add(new ROC(steps));
            }

            ROC single = new ROC(steps);
            for (int i = 0; i < nArrays; i++) {
                INDArray p = Nd4j.rand(minibatch, 2);
                p.diviColumnVector(p.sum(1));

                INDArray l = Nd4j.zeros(minibatch, 2);
                for (int j = 0; j < minibatch; j++) {
                    l.putScalar(j, r.nextInt(2), 1.0);
                }

                single.eval(l, p);

                ROC other = rocList.get(i % rocList.size());
                other.eval(l, p);
            }

            ROC first = rocList.get(0);
            for (int i = 1; i < nROCs; i++) {
                first.merge(rocList.get(i));
            }

            double singleAUC = single.calculateAUC();
            assertTrue(singleAUC >= 0.0 && singleAUC <= 1.0);
            assertEquals(singleAUC, first.calculateAUC(), 1e-6);

            assertEquals(single.getRocCurve(), first.getRocCurve());
        }
    }

    @Test
    public void testROCMerging2() {
        int nArrays = 10;
        int minibatch = 64;
        int exactAllocBlockSize = 10;
        int nROCs = 3;
        int steps = 0;  //Exact

        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);

        List<ROC> rocList = new ArrayList<>();
        for (int i = 0; i < nROCs; i++) {
            rocList.add(new ROC(steps, true, exactAllocBlockSize));
        }

        ROC single = new ROC(steps);
        for (int i = 0; i < nArrays; i++) {
            INDArray p = Nd4j.rand(minibatch, 2);
            p.diviColumnVector(p.sum(1));

            INDArray l = Nd4j.zeros(minibatch, 2);
            for (int j = 0; j < minibatch; j++) {
                l.putScalar(j, r.nextInt(2), 1.0);
            }

            single.eval(l, p);

            ROC other = rocList.get(i % rocList.size());
            other.eval(l, p);
        }

        ROC first = rocList.get(0);
        for (int i = 1; i < nROCs; i++) {
            first.merge(rocList.get(i));
        }

        double singleAUC = single.calculateAUC();
        assertTrue(singleAUC >= 0.0 && singleAUC <= 1.0);
        assertEquals(singleAUC, first.calculateAUC(), 1e-6);

        assertEquals(single.getRocCurve(), first.getRocCurve());
    }


    @Test
    public void testROCMultiMerging() {

        int nArrays = 10;
        int minibatch = 64;
        int nROCs = 3;
        int nClasses = 3;

        for (int steps : new int[] {20, 0}) { //0 steps: exact
            //            int steps = 20;

            Nd4j.getRandom().setSeed(12345);
            Random r = new Random(12345);

            List<ROCMultiClass> rocList = new ArrayList<>();
            for (int i = 0; i < nROCs; i++) {
                rocList.add(new ROCMultiClass(steps));
            }

            ROCMultiClass single = new ROCMultiClass(steps);
            for (int i = 0; i < nArrays; i++) {
                INDArray p = Nd4j.rand(minibatch, nClasses);
                p.diviColumnVector(p.sum(1));

                INDArray l = Nd4j.zeros(minibatch, nClasses);
                for (int j = 0; j < minibatch; j++) {
                    l.putScalar(j, r.nextInt(nClasses), 1.0);
                }

                single.eval(l, p);

                ROCMultiClass other = rocList.get(i % rocList.size());
                other.eval(l, p);
            }

            ROCMultiClass first = rocList.get(0);
            for (int i = 1; i < nROCs; i++) {
                first.merge(rocList.get(i));
            }

            for (int i = 0; i < nClasses; i++) {
                assertEquals(single.calculateAUC(i), first.calculateAUC(i), 1e-6);

                assertEquals(single.getRocCurve(i), first.getRocCurve(i));
            }
        }
    }

    @Test
    public void RocEvalSanityCheck() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).seed(12345)
                        .list()
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

        iter.setPreProcessor(ns);

        for (int i = 0; i < 10; i++) {
            net.fit(ds);
        }

        for (int steps : new int[] {32, 0}) { //Steps = 0: exact
            System.out.println("steps: " + steps);

            iter.reset();
            ds = iter.next();
            INDArray f = ds.getFeatures();
            INDArray l = ds.getLabels();
            INDArray out = net.output(f);
            //            System.out.println(f);
            //            System.out.println(out);
            ROCMultiClass manual = new ROCMultiClass(steps);
            manual.eval(l, out);

            iter.reset();
            ROCMultiClass roc = net.evaluateROCMultiClass(iter, steps);


            for (int i = 0; i < 3; i++) {
                double rocExp = manual.calculateAUC(i);
                double rocAct = roc.calculateAUC(i);
                assertEquals(rocExp, rocAct, 1e-6);

                RocCurve rc = roc.getRocCurve(i);
                RocCurve rm = manual.getRocCurve(i);

                assertEquals(rc, rm);
            }
        }
    }

    @Test
    public void testAUCPrecisionRecall() {
        //Assume 2 positive examples, at 0.33 and 0.66 predicted, 1 negative example at 0.25 prob
        //at threshold 0 to 0.24999: tp=2, fp=1, fn=0, tn=0 prec=2/(2+1)=0.666, recall=2/2=1.0
        //at threshold 0.25 to 0.33: tp=2, fp=0, fn=0, tn=1 prec=2/2=1, recall=2/2=1
        //at threshold 0.331 to 0.66: tp=1, fp=0, fn=1, tn=1 prec=1/1=1, recall=1/2=0.5
        //at threshold 0.661 to 1.0:  tp=0, fp=0, fn=2, tn=1 prec=0/0=1, recall=0/2=0

        for (int steps : new int[] {10, 0}) { //0 steps = exact
            //        for (int steps : new int[] {0}) { //0 steps = exact
            String msg = "Steps = " + steps;
            //area: 1.0
            ROC r = new ROC(steps);
            INDArray zero = Nd4j.zeros(1);
            INDArray one = Nd4j.ones(1);
            r.eval(zero, Nd4j.create(new double[] {0.25}));
            r.eval(one, Nd4j.create(new double[] {0.33}));
            r.eval(one, Nd4j.create(new double[] {0.66}));

            PrecisionRecallCurve prc = r.getPrecisionRecallCurve();

            double auprc = r.calculateAUCPR();
            assertEquals(msg, 1.0, auprc, 1e-6);

            //Assume 2 positive examples, at 0.33 and 0.66 predicted, 1 negative example at 0.5 prob
            //at threshold 0 to 0.33: tp=2, fp=1, fn=0, tn=0 prec=2/(2+1)=0.666, recall=2/2=1.0
            //at threshold 0.331 to 0.5: tp=1, fp=1, fn=1, tn=0 prec=1/2=0.5, recall=1/2=0.5
            //at threshold 0.51 to 0.66: tp=1, fp=0, fn=1, tn=1 prec=1/1=1, recall=1/2=0.5
            //at threshold 0.661 to 1.0:  tp=0, fp=0, fn=2, tn=1 prec=0/0=1, recall=0/2=0
            //Area: 0.5 + 0.25 + 0.5*0.5*(0.66666-0.5) = 0.5+0.25+0.04165 = 0.7916666666667
            //But, we use 10 steps so the calculation might not match this exactly, but should be close
            r = new ROC(steps);
            r.eval(one, Nd4j.create(new double[] {0.33}));
            r.eval(zero, Nd4j.create(new double[] {0.5}));
            r.eval(one, Nd4j.create(new double[] {0.66}));

            double precision;
            if (steps == 0) {
                precision = 1e-8;
            } else {
                precision = 1e-4;
            }
            assertEquals(msg, 0.7916666666667, r.calculateAUCPR(), precision);
        }
    }


    @Test
    public void testRocAucExact() {

        //Check the implementation vs. Scikitlearn
        /*
        np.random.seed(12345)
        prob = np.random.rand(30,1)
        label = np.random.randint(0,2,(30,1))
        positiveClass = 1;
        
        fpr, tpr, thr = sklearn.metrics.roc_curve(label, prob, positiveClass, None, False)
        auc = sklearn.metrics.auc(fpr, tpr)
        
        #PR curve
        p, r, t = precision_recall_curve(label, prob, positiveClass)
        
        #sklearn.metrics.average_precision_score: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        # "This score corresponds to the area under the precision-recall curve."
        auprc = sklearn.metrics.average_precision_score(label, prob)
        print(auprc)
        
        fpr
        [ 0.          0.15789474  0.15789474  0.31578947  0.31578947  0.52631579
          0.52631579  0.68421053  0.68421053  0.84210526  0.84210526  0.89473684
          0.89473684  1.        ]
        tpr
        [ 0.09090909  0.09090909  0.18181818  0.18181818  0.36363636  0.36363636
          0.45454545  0.45454545  0.72727273  0.72727273  0.90909091  0.90909091
          1.          1.        ]
        threshold
        [ 0.99401459  0.96130674  0.92961609  0.79082252  0.74771481  0.67687371
          0.65641118  0.64247533  0.46759901  0.31637555  0.20456028  0.18391881
          0.17091426  0.0083883 ]
        
        p, r, t = precision_recall_curve(label, prob)
        
        Precision
        [ 0.39285714  0.37037037  0.38461538  0.36        0.33333333  0.34782609
          0.36363636  0.38095238  0.35        0.31578947  0.27777778  0.29411765
          0.3125      0.33333333  0.28571429  0.30769231  0.33333333  0.36363636
          0.4         0.33333333  0.25        0.28571429  0.33333333  0.4         0.25
          0.33333333  0.5         1.          1.        ]
        Recall
        [ 1.          0.90909091  0.90909091  0.81818182  0.72727273  0.72727273
          0.72727273  0.72727273  0.63636364  0.54545455  0.45454545  0.45454545
          0.45454545  0.45454545  0.36363636  0.36363636  0.36363636  0.36363636
          0.36363636  0.27272727  0.18181818  0.18181818  0.18181818  0.18181818
          0.09090909  0.09090909  0.09090909  0.09090909  0.        ]
        Threshold
        [ 0.17091426  0.18391881  0.20456028  0.29870371  0.31637555  0.32558468
          0.43964461  0.46759901  0.56772503  0.5955447   0.64247533  0.6531771
          0.65356987  0.65641118  0.67687371  0.71745362  0.72368535  0.72968908
          0.74771481  0.74890664  0.79082252  0.80981255  0.87217591  0.92961609
          0.96130674  0.96451452  0.9646476   0.99401459]
        
        AUPRC
        0.398963619227
         */

        double[] p = new double[] {0.92961609, 0.31637555, 0.18391881, 0.20456028, 0.56772503, 0.5955447, 0.96451452,
                        0.6531771, 0.74890664, 0.65356987, 0.74771481, 0.96130674, 0.0083883, 0.10644438, 0.29870371,
                        0.65641118, 0.80981255, 0.87217591, 0.9646476, 0.72368535, 0.64247533, 0.71745362, 0.46759901,
                        0.32558468, 0.43964461, 0.72968908, 0.99401459, 0.67687371, 0.79082252, 0.17091426};

        double[] l = new double[] {1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                        0, 1};

        double[] fpr_skl = new double[] {0.0, 0.0, 0.15789474, 0.15789474, 0.31578947, 0.31578947, 0.52631579,
                        0.52631579, 0.68421053, 0.68421053, 0.84210526, 0.84210526, 0.89473684, 0.89473684, 1.0};
        double[] tpr_skl = new double[] {0.0, 0.09090909, 0.09090909, 0.18181818, 0.18181818, 0.36363636, 0.36363636,
                        0.45454545, 0.45454545, 0.72727273, 0.72727273, 0.90909091, 0.90909091, 1.0, 1.0};
        //Note the change to the last value: same TPR and FPR at 0.0083883 and 0.0 -> we add the 0.0 threshold edge case + combine with the previous one. Same result
        double[] thr_skl = new double[] {1.0, 0.99401459, 0.96130674, 0.92961609, 0.79082252, 0.74771481, 0.67687371,
                        0.65641118, 0.64247533, 0.46759901, 0.31637555, 0.20456028, 0.18391881, 0.17091426, 0.0};

        INDArray prob = Nd4j.create(p, new int[] {30, 1});
        INDArray label = Nd4j.create(l, new int[] {30, 1});

        ROC roc = new ROC(0);
        roc.eval(label, prob);

        RocCurve rocCurve = roc.getRocCurve();

        //        System.out.println("Thr: " + Arrays.toString(rocCurve[0]));
        //        System.out.println("FPR: " + Arrays.toString(rocCurve[1]));
        //        System.out.println("TPR: " + Arrays.toString(rocCurve[2]));
        //        System.out.println("AUC: " + roc.calculateAUC());

        assertArrayEquals(thr_skl, rocCurve.getThreshold(), 1e-6);
        assertArrayEquals(fpr_skl, rocCurve.getFpr(), 1e-6);
        assertArrayEquals(tpr_skl, rocCurve.getTpr(), 1e-6);

        double auc = roc.calculateAUC();
        double aucExpSKL = 0.459330143541;
        assertEquals(aucExpSKL, auc, 1e-6);

        roc = new ROC(0, false);
        roc.eval(label, prob);
        assertEquals(aucExpSKL, roc.calculateAUC(), 1e-6);



        //Check area under PR curve
        roc = new ROC(0, true);
        roc.eval(label, prob);

        //Unfortunately some of the sklearn points are redundant... and they are missing the edge cases.
        // so a direct element-by-element comparison is not possible, unlike in the ROC case

        double auprcExp = 0.398963619227;
        double auprcAct = roc.calculateAUCPR();
        assertEquals(auprcExp, auprcAct, 1e-8);

        roc = new ROC(0, false);
        roc.eval(label, prob);
        assertEquals(auprcExp, roc.calculateAUCPR(), 1e-8);


        //Check precision recall curve counts etc
        PrecisionRecallCurve prc = roc.getPrecisionRecallCurve();
        for (int i = 0; i < thr_skl.length; i++) {
            double threshold = thr_skl[i] - 1e-6; //Subtract a bit, so we get the correct point (rounded up on the get op)
            threshold = Math.max(0.0, threshold);
            PrecisionRecallCurve.Confusion c = prc.getConfusionMatrixAtThreshold(threshold);
            int tp = c.getTpCount();
            int fp = c.getFpCount();
            int tn = c.getTnCount();
            int fn = c.getFnCount();

            assertEquals(30, tp + fp + tn + fn);

            double prec = tp / (double) (tp + fp);
            double rec = tp / (double) (tp + fn);
            double fpr = fp / 19.0;

            if (c.getPoint().getThreshold() == 0.0) {
                rec = 1.0;
                prec = 11.0 / 30; //11 positives, 30 total
            } else if (c.getPoint().getThreshold() == 1.0) {
                rec = 0.0;
                prec = 1.0;
            }

            //            System.out.println(i + "\t" + threshold);
            assertEquals(tpr_skl[i], rec, 1e-6);
            assertEquals(fpr_skl[i], fpr, 1e-6);

            assertEquals(rec, c.getPoint().getRecall(), 1e-6);
            assertEquals(prec, c.getPoint().getPrecision(), 1e-6);
        }


        //Check edge case: perfect classifier
        prob = Nd4j.create(new double[] {0.1, 0.2, 0.5, 0.9}, new int[] {4, 1});
        label = Nd4j.create(new double[] {0, 0, 1, 1}, new int[] {4, 1});
        roc = new ROC(0);
        roc.eval(label, prob);
        assertEquals(1.0, roc.calculateAUC(), 1e-8);

        assertEquals(1.0, roc.calculateAUCPR(), 1e-8);
    }


    @Test
    public void rocExactEdgeCaseReallocation() {

        //Set reallocation block size to say 20, but then evaluate a 100-length array

        ROC roc = new ROC(0, true, 50);

        roc.eval(Nd4j.rand(100, 1), Nd4j.ones(100, 1));

    }


    @Test
    public void testPrecisionRecallCurveGetPointMethods() {
        double[] threshold = new double[101];
        double[] precision = threshold;
        double[] recall = new double[101];
        int i = 0;
        for (double d = 0; d <= 1; d += 0.01) {
            threshold[i] = d;
            recall[i] = 1.0 - d;
            i++;
        }


        PrecisionRecallCurve prc = new PrecisionRecallCurve(threshold, precision, recall, null, null, null, -1);

        PrecisionRecallCurve.Point[] points = new PrecisionRecallCurve.Point[] {
                        //Test exact:
                        prc.getPointAtThreshold(0.05), prc.getPointAtPrecision(0.05), prc.getPointAtRecall(1 - 0.05),

                        //Test approximate (point doesn't exist exactly). When it doesn't exist:
                        //Threshold: lowest threshold equal to or exceeding the specified threshold value
                        //Precision: lowest threshold equal to or exceeding the specified precision value
                        //Recall: highest threshold equal to or exceeding the specified recall value
                        prc.getPointAtThreshold(0.0495), prc.getPointAtPrecision(0.0495),
                        prc.getPointAtRecall(1 - 0.0505)};



        for (PrecisionRecallCurve.Point p : points) {
            assertEquals(5, p.getIdx());
            assertEquals(0.05, p.getThreshold(), 1e-6);
            assertEquals(0.05, p.getPrecision(), 1e-6);
            assertEquals(1 - 0.05, p.getRecall(), 1e-6);
        }
    }

    @Test
    public void testPrecisionRecallCurveConfusion() {
        //Sanity check: values calculated from the confusion matrix should match the PR curve values

        for (boolean removeRedundantPts : new boolean[] {true, false}) {
            ROC r = new ROC(0, removeRedundantPts);

            INDArray labels = Nd4j.getExecutioner()
                            .exec(new BernoulliDistribution(Nd4j.createUninitialized(100, 1), 0.5));
            INDArray probs = Nd4j.rand(100, 1);

            r.eval(labels, probs);

            PrecisionRecallCurve prc = r.getPrecisionRecallCurve();
            int nPoints = prc.numPoints();

            for (int i = 0; i < nPoints; i++) {
                PrecisionRecallCurve.Confusion c = prc.getConfusionMatrixAtPoint(i);
                PrecisionRecallCurve.Point p = c.getPoint();

                int tp = c.getTpCount();
                int fp = c.getFpCount();
                int fn = c.getFnCount();

                double prec = tp / (double) (tp + fp);
                double rec = tp / (double) (tp + fn);

                //Handle edge cases:
                if (tp == 0 && fp == 0) {
                    prec = 1.0;
                }

                assertEquals(p.getPrecision(), prec, 1e-8);
                assertEquals(p.getRecall(), rec, 1e-8);
            }
        }
    }


    @Test
    public void testRocMerge(){
        Nd4j.getRandom().setSeed(12345);

        ROC roc = new ROC();
        ROC roc1 = new ROC();
        ROC roc2 = new ROC();

        int nOut = 2;

        Random r = new Random(12345);
        for( int i=0; i<10; i++ ){
            INDArray labels = Nd4j.zeros(3, nOut);
            for( int j=0; j<3; j++ ){
                labels.putScalar(j, r.nextInt(nOut), 1.0 );
            }
            INDArray out = Nd4j.rand(3, nOut);
            out.diviColumnVector(out.sum(1));

            roc.eval(labels, out);
            if(i % 2 == 0){
                roc1.eval(labels, out);
            } else {
                roc2.eval(labels, out);
            }
        }

        roc1.calculateAUC();
        roc1.calculateAUCPR();
        roc2.calculateAUC();
        roc2.calculateAUCPR();

        roc1.merge(roc2);

        double aucExp = roc.calculateAUC();
        double auprc = roc.calculateAUCPR();

        double aucAct = roc1.calculateAUC();
        double auprcAct = roc1.calculateAUCPR();

        assertEquals(aucExp, aucAct, 1e-6);
        assertEquals(auprc, auprcAct, 1e-6);
    }

    @Test
    public void testRocMultiMerge(){
        Nd4j.getRandom().setSeed(12345);

        ROCMultiClass roc = new ROCMultiClass();
        ROCMultiClass roc1 = new ROCMultiClass();
        ROCMultiClass roc2 = new ROCMultiClass();

        int nOut = 5;

        Random r = new Random(12345);
        for( int i=0; i<10; i++ ){
            INDArray labels = Nd4j.zeros(3, nOut);
            for( int j=0; j<3; j++ ){
                labels.putScalar(j, r.nextInt(nOut), 1.0 );
            }
            INDArray out = Nd4j.rand(3, nOut);
            out.diviColumnVector(out.sum(1));

            roc.eval(labels, out);
            if(i % 2 == 0){
                roc1.eval(labels, out);
            } else {
                roc2.eval(labels, out);
            }
        }

        for( int i=0; i<nOut; i++ ) {
            roc1.calculateAUC(i);
            roc1.calculateAUCPR(i);
            roc2.calculateAUC(i);
            roc2.calculateAUCPR(i);
        }

        roc1.merge(roc2);

        for( int i=0; i<nOut; i++ ) {

            double aucExp = roc.calculateAUC(i);
            double auprc = roc.calculateAUCPR(i);

            double aucAct = roc1.calculateAUC(i);
            double auprcAct = roc1.calculateAUCPR(i);

            assertEquals(aucExp, aucAct, 1e-6);
            assertEquals(auprc, auprcAct, 1e-6);
        }
    }

    @Test
    public void testRocBinaryMerge(){
        Nd4j.getRandom().setSeed(12345);

        ROCBinary roc = new ROCBinary();
        ROCBinary roc1 = new ROCBinary();
        ROCBinary roc2 = new ROCBinary();

        int nOut = 5;

        for( int i=0; i<10; i++ ){
            INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(3, nOut),0.5));
            INDArray out = Nd4j.rand(3, nOut);
            out.diviColumnVector(out.sum(1));

            roc.eval(labels, out);
            if(i % 2 == 0){
                roc1.eval(labels, out);
            } else {
                roc2.eval(labels, out);
            }
        }

        for( int i=0; i<nOut; i++ ) {
            roc1.calculateAUC(i);
            roc1.calculateAUCPR(i);
            roc2.calculateAUC(i);
            roc2.calculateAUCPR(i);
        }

        roc1.merge(roc2);

        for( int i=0; i<nOut; i++ ) {

            double aucExp = roc.calculateAUC(i);
            double auprc = roc.calculateAUCPR(i);

            double aucAct = roc1.calculateAUC(i);
            double auprcAct = roc1.calculateAUCPR(i);

            assertEquals(aucExp, aucAct, 1e-6);
            assertEquals(auprc, auprcAct, 1e-6);
        }
    }

}
