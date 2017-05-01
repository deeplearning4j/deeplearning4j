package org.deeplearning4j.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 21/03/2017.
 */
public class ROCBinaryTest {

    @Test
    public void testROCBinary() {
        //Compare ROCBinary to ROC class

        Nd4j.getRandom().setSeed(12345);

        int nExamples = 50;
        int nOut = 4;
        int[] shape = {nExamples, nOut};

        int thresholdSteps = 30;

        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape), 0.5));

        INDArray predicted = Nd4j.rand(shape);
        INDArray binaryPredicted = predicted.gt(0.5);

        ROCBinary rb = new ROCBinary(thresholdSteps);
        rb.eval(labels, predicted);

        System.out.println(rb.stats());

        double eps = 1e-6;
        for (int i = 0; i < nOut; i++) {
            INDArray lCol = labels.getColumn(i);
            INDArray pCol = predicted.getColumn(i);


            ROC r = new ROC(thresholdSteps);
            r.eval(lCol, pCol);

            double aucExp = r.calculateAUC();
            double auc = rb.calculateAUC(i);

            assertEquals(aucExp, auc, eps);

            long apExp = r.getCountActualPositive();
            long ap = rb.getCountActualPositive(i);
            assertEquals(ap, apExp);

            long anExp = r.getCountActualNegative();
            long an = rb.getCountActualNegative(i);
            assertEquals(anExp, an);

            List<ROC.PrecisionRecallPoint> pExp = r.getPrecisionRecallCurve();
            List<ROCBinary.PrecisionRecallPoint> p = rb.getPrecisionRecallCurve(i);
            assertEquals(pExp.size(), p.size());

            for (int j = 0; j < pExp.size(); j++) {
                ROC.PrecisionRecallPoint a = pExp.get(j);
                ROCBinary.PrecisionRecallPoint b = p.get(j);
                assertEquals(a.getClassiferThreshold(), b.getClassiferThreshold(), eps);
                assertEquals(a.getPrecision(), b.getPrecision(), eps);
                assertEquals(a.getRecall(), b.getRecall(), eps);
            }

            double[][] d1 = r.getResultsAsArray();
            double[][] d2 = rb.getResultsAsArray(i);

            assertEquals(d1.length, d2.length);
            for (int j = 0; j < d1.length; j++) {
                assertArrayEquals(d1[j], d2[j], eps);
            }
        }
    }

    @Test
    public void testRocBinaryMerging() {
        int nSteps = 30;
        int nOut = 4;
        int[] shape1 = {30, nOut};
        int[] shape2 = {50, nOut};

        Nd4j.getRandom().setSeed(12345);
        INDArray l1 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape1), 0.5));
        INDArray l2 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape2), 0.5));
        INDArray p1 = Nd4j.rand(shape1);
        INDArray p2 = Nd4j.rand(shape2);

        ROCBinary rb = new ROCBinary(nSteps);
        rb.eval(l1, p1);
        rb.eval(l2, p2);

        ROCBinary rb1 = new ROCBinary(nSteps);
        rb1.eval(l1, p1);

        ROCBinary rb2 = new ROCBinary(nSteps);
        rb2.eval(l2, p2);

        rb1.merge(rb2);

        assertEquals(rb.stats(), rb1.stats());
    }


    @Test
    public void testROCBinaryPerOutputMasking() {
        int nSteps = 30;

        //Here: we'll create a test array, then insert some 'masked out' values, and ensure we get the same results
        INDArray mask = Nd4j.create(new double[][] {{1, 1, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}});

        INDArray labels = Nd4j.create(new double[][] {{0, 1, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1}});

        //Remove the 1 masked value for each column
        INDArray labelsExMasked = Nd4j.create(new double[][] {{0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}});

        INDArray predicted = Nd4j.create(new double[][] {{0.9, 0.4, 0.6}, {0.2, 0.8, 0.4}, {0.6, 0.1, 0.1},
                        {0.3, 0.7, 0.2}, {0.8, 0.6, 0.6}});

        INDArray predictedExMasked = Nd4j
                        .create(new double[][] {{0.9, 0.4, 0.6}, {0.6, 0.8, 0.4}, {0.3, 0.7, 0.1}, {0.8, 0.6, 0.6}});

        ROCBinary rbMasked = new ROCBinary(nSteps);
        rbMasked.eval(labels, predicted, mask);

        ROCBinary rb = new ROCBinary(nSteps);
        rb.eval(labelsExMasked, predictedExMasked);

        assertEquals(rb.stats(), rbMasked.stats());

        for (int i = 0; i < 3; i++) {
            List<ROCBinary.PrecisionRecallPoint> pExp = rb.getPrecisionRecallCurve(i);
            List<ROCBinary.PrecisionRecallPoint> p = rbMasked.getPrecisionRecallCurve(i);

            assertEquals(pExp, p);
        }
    }

}
