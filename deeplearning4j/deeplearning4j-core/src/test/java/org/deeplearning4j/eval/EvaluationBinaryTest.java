package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 20/03/2017.
 */
public class EvaluationBinaryTest extends BaseDL4JTest {

    @Test
    public void testEvaluationBinary() {
        //Compare EvaluationBinary to Evaluation class

        Nd4j.getRandom().setSeed(12345);

        int nExamples = 50;
        int nOut = 4;
        int[] shape = {nExamples, nOut};

        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape), 0.5));

        INDArray predicted = Nd4j.rand(shape);
        INDArray binaryPredicted = predicted.gt(0.5);

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(labels, predicted);

        System.out.println(eb.stats());

        double eps = 1e-6;
        for (int i = 0; i < nOut; i++) {
            INDArray lCol = labels.getColumn(i);
            INDArray pCol = predicted.getColumn(i);
            INDArray bpCol = binaryPredicted.getColumn(i);

            int countCorrect = 0;
            int tpCount = 0;
            int tnCount = 0;
            for (int j = 0; j < lCol.length(); j++) {
                if (lCol.getDouble(j) == bpCol.getDouble(j)) {
                    countCorrect++;
                    if (lCol.getDouble(j) == 1) {
                        tpCount++;
                    } else {
                        tnCount++;
                    }
                }
            }
            double acc = countCorrect / (double) lCol.length();

            Evaluation e = new Evaluation();
            e.eval(lCol, pCol);

            assertEquals(acc, eb.accuracy(i), eps);
            assertEquals(e.accuracy(), eb.accuracy(i), eps);
            assertEquals(e.precision(1), eb.precision(i), eps);
            assertEquals(e.recall(1), eb.recall(i), eps);
            assertEquals(e.f1(1), eb.f1(i), eps);

            assertEquals(tpCount, eb.truePositives(i));
            assertEquals(tnCount, eb.trueNegatives(i));

            assertEquals((int) e.truePositives().get(1), eb.truePositives(i));
            assertEquals((int) e.trueNegatives().get(1), eb.trueNegatives(i));
            assertEquals((int) e.falsePositives().get(1), eb.falsePositives(i));
            assertEquals((int) e.falseNegatives().get(1), eb.falseNegatives(i));

            assertEquals(nExamples, eb.totalCount(i));
        }
    }

    @Test
    public void testEvaluationBinaryMerging() {
        int nOut = 4;
        int[] shape1 = {30, nOut};
        int[] shape2 = {50, nOut};

        Nd4j.getRandom().setSeed(12345);
        INDArray l1 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape1), 0.5));
        INDArray l2 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape2), 0.5));
        INDArray p1 = Nd4j.rand(shape1);
        INDArray p2 = Nd4j.rand(shape2);

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(l1, p1);
        eb.eval(l2, p2);

        EvaluationBinary eb1 = new EvaluationBinary();
        eb1.eval(l1, p1);

        EvaluationBinary eb2 = new EvaluationBinary();
        eb2.eval(l2, p2);

        eb1.merge(eb2);

        assertEquals(eb.stats(), eb1.stats());
    }

    @Test
    public void testEvaluationBinaryPerOutputMasking() {

        //Provide a mask array: "ignore" the masked steps

        INDArray mask = Nd4j.create(new double[][] {{1, 1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 0}, {1, 1, 1}});

        INDArray labels = Nd4j.create(new double[][] {{1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 1}});

        INDArray predicted = Nd4j.create(new double[][] {{0.9, 0.9, 0.9}, {0.7, 0.7, 0.7}, {0.6, 0.6, 0.6},
                        {0.4, 0.4, 0.4}, {0.1, 0.1, 0.1}});

        //Correct?
        //      Y Y m
        //      N m m
        //      Y Y m
        //      Y m m
        //      N Y N

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(labels, predicted, mask);

        assertEquals(0.6, eb.accuracy(0), 1e-6);
        assertEquals(1.0, eb.accuracy(1), 1e-6);
        assertEquals(0.0, eb.accuracy(2), 1e-6);

        assertEquals(2, eb.truePositives(0));
        assertEquals(2, eb.truePositives(1));
        assertEquals(0, eb.truePositives(2));

        assertEquals(1, eb.trueNegatives(0));
        assertEquals(1, eb.trueNegatives(1));
        assertEquals(0, eb.trueNegatives(2));

        assertEquals(1, eb.falsePositives(0));
        assertEquals(0, eb.falsePositives(1));
        assertEquals(0, eb.falsePositives(2));

        assertEquals(1, eb.falseNegatives(0));
        assertEquals(0, eb.falseNegatives(1));
        assertEquals(1, eb.falseNegatives(2));
    }

    @Test
    public void testTimeSeriesEval() {

        int[] shape = {2, 4, 3};
        Nd4j.getRandom().setSeed(12345);
        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape), 0.5));
        INDArray predicted = Nd4j.rand(shape);
        INDArray mask = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(shape), 0.5));

        EvaluationBinary eb1 = new EvaluationBinary();
        eb1.eval(labels, predicted, mask);

        EvaluationBinary eb2 = new EvaluationBinary();
        for (int i = 0; i < shape[2]; i++) {
            INDArray l = labels.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));
            INDArray p = predicted.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));
            INDArray m = mask.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i));

            eb2.eval(l, p, m);
        }

        assertEquals(eb2.stats(), eb1.stats());
    }

    @Test
    public void testEvaluationBinaryWithROC() {
        //Simple test for nested ROCBinary in EvaluationBinary

        Nd4j.getRandom().setSeed(12345);
        INDArray l1 = Nd4j.getExecutioner()
                        .exec(new BernoulliDistribution(Nd4j.createUninitialized(new int[] {50, 4}), 0.5));
        INDArray p1 = Nd4j.rand(50, 4);

        EvaluationBinary eb = new EvaluationBinary(4, 30);
        eb.eval(l1, p1);

        System.out.println(eb.stats());
    }
}
