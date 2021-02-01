/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.evaluation;

import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.nd4j.evaluation.classification.EvaluationBinary.Metric.*;
/**
 * Created by Alex on 20/03/2017.
 */
public class EvaluationBinaryTest extends BaseNd4jTest {

    public EvaluationBinaryTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testEvaluationBinary() {
        //Compare EvaluationBinary to Evaluation class
        DataType dtypeBefore = Nd4j.defaultFloatingPointType();
        EvaluationBinary first = null;
        String sFirst = null;
        try {
            for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT}) {
                Nd4j.setDefaultDataTypes(globalDtype, globalDtype.isFPType() ? globalDtype : DataType.DOUBLE);
                for (DataType lpDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {

                    Nd4j.getRandom().setSeed(12345);

                    int nExamples = 50;
                    int nOut = 4;
                    long[] shape = {nExamples, nOut};

                    INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(lpDtype, shape), 0.5));

                    INDArray predicted = Nd4j.rand(lpDtype, shape);
                    INDArray binaryPredicted = predicted.gt(0.5);

                    EvaluationBinary eb = new EvaluationBinary();
                    eb.eval(labels, predicted);

                    //System.out.println(eb.stats());

                    double eps = 1e-6;
                    for (int i = 0; i < nOut; i++) {
                        INDArray lCol = labels.getColumn(i,true);
                        INDArray pCol = predicted.getColumn(i,true);
                        INDArray bpCol = binaryPredicted.getColumn(i,true);

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
                        assertEquals(e.accuracy(), eb.scoreForMetric(ACCURACY, i), eps);
                        assertEquals(e.precision(1), eb.scoreForMetric(PRECISION, i), eps);
                        assertEquals(e.recall(1), eb.scoreForMetric(RECALL, i), eps);
                        assertEquals(e.f1(1), eb.scoreForMetric(F1, i), eps);
                        assertEquals(e.falseAlarmRate(), eb.scoreForMetric(FAR, i), eps);
                        assertEquals(e.falsePositiveRate(1), eb.falsePositiveRate(i), eps);


                        assertEquals(tpCount, eb.truePositives(i));
                        assertEquals(tnCount, eb.trueNegatives(i));

                        assertEquals((int) e.truePositives().get(1), eb.truePositives(i));
                        assertEquals((int) e.trueNegatives().get(1), eb.trueNegatives(i));
                        assertEquals((int) e.falsePositives().get(1), eb.falsePositives(i));
                        assertEquals((int) e.falseNegatives().get(1), eb.falseNegatives(i));

                        assertEquals(nExamples, eb.totalCount(i));

                        String s = eb.stats();
                        if(first == null) {
                            first = eb;
                            sFirst = s;
                        } else {
                            assertEquals(first, eb);
                            assertEquals(sFirst, s);
                        }
                    }
                }
            }
        } finally {
            Nd4j.setDefaultDataTypes(dtypeBefore, dtypeBefore);
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

//        System.out.println(eb.stats());
        eb.stats();
    }


    @Test
    public void testEvaluationBinary3d() {
        INDArray prediction = Nd4j.rand(DataType.FLOAT, 2, 5, 10);
        INDArray label = Nd4j.rand(DataType.FLOAT, 2, 5, 10);


        List<INDArray> rowsP = new ArrayList<>();
        List<INDArray> rowsL = new ArrayList<>();
        NdIndexIterator iter = new NdIndexIterator(2, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1])};
            rowsP.add(prediction.get(idxs));
            rowsL.add(label.get(idxs));
        }

        INDArray p2d = Nd4j.vstack(rowsP);
        INDArray l2d = Nd4j.vstack(rowsL);

        EvaluationBinary e3d = new EvaluationBinary();
        EvaluationBinary e2d = new EvaluationBinary();

        e3d.eval(label, prediction);
        e2d.eval(l2d, p2d);

        for (EvaluationBinary.Metric m : EvaluationBinary.Metric.values()) {
            for( int i=0; i<5; i++ ) {
                double d1 = e3d.scoreForMetric(m, i);
                double d2 = e2d.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testEvaluationBinary4d() {
        INDArray prediction = Nd4j.rand(DataType.FLOAT, 2, 3, 10, 10);
        INDArray label = Nd4j.rand(DataType.FLOAT, 2, 3, 10, 10);


        List<INDArray> rowsP = new ArrayList<>();
        List<INDArray> rowsL = new ArrayList<>();
        NdIndexIterator iter = new NdIndexIterator(2, 10, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1]), NDArrayIndex.point(idx[2])};
            rowsP.add(prediction.get(idxs));
            rowsL.add(label.get(idxs));
        }

        INDArray p2d = Nd4j.vstack(rowsP);
        INDArray l2d = Nd4j.vstack(rowsL);

        EvaluationBinary e4d = new EvaluationBinary();
        EvaluationBinary e2d = new EvaluationBinary();

        e4d.eval(label, prediction);
        e2d.eval(l2d, p2d);

        for (EvaluationBinary.Metric m : EvaluationBinary.Metric.values()) {
            for( int i=0; i<3; i++ ) {
                double d1 = e4d.scoreForMetric(m, i);
                double d2 = e2d.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testEvaluationBinary3dMasking() {
        INDArray prediction = Nd4j.rand(DataType.FLOAT, 2, 3, 10);
        INDArray label = Nd4j.rand(DataType.FLOAT, 2, 3, 10);

        List<INDArray> rowsP = new ArrayList<>();
        List<INDArray> rowsL = new ArrayList<>();

        //Check "DL4J-style" 2d per timestep masking [minibatch, seqLength] mask shape
        INDArray mask2d = Nd4j.randomBernoulli(0.5, 2, 10);
        rowsP.clear();
        rowsL.clear();
        NdIndexIterator iter = new NdIndexIterator(2, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            if(mask2d.getDouble(idx[0], idx[1]) != 0.0) {
                INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1])};
                rowsP.add(prediction.get(idxs));
                rowsL.add(label.get(idxs));
            }
        }
        INDArray p2d = Nd4j.vstack(rowsP);
        INDArray l2d = Nd4j.vstack(rowsL);

        EvaluationBinary e3d_m2d = new EvaluationBinary();
        EvaluationBinary e2d_m2d = new EvaluationBinary();
        e3d_m2d.eval(label, prediction, mask2d);
        e2d_m2d.eval(l2d, p2d);



        //Check per-output masking:
        INDArray perOutMask = Nd4j.randomBernoulli(0.5, label.shape());
        rowsP.clear();
        rowsL.clear();
        List<INDArray> rowsM = new ArrayList<>();
        iter = new NdIndexIterator(2, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1])};
            rowsP.add(prediction.get(idxs));
            rowsL.add(label.get(idxs));
            rowsM.add(perOutMask.get(idxs));
        }
        p2d = Nd4j.vstack(rowsP);
        l2d = Nd4j.vstack(rowsL);
        INDArray m2d = Nd4j.vstack(rowsM);

        EvaluationBinary e4d_m2 = new EvaluationBinary();
        EvaluationBinary e2d_m2 = new EvaluationBinary();
        e4d_m2.eval(label, prediction, perOutMask);
        e2d_m2.eval(l2d, p2d, m2d);
        for(EvaluationBinary.Metric m : EvaluationBinary.Metric.values()){
            for(int i=0; i<3; i++ ) {
                double d1 = e4d_m2.scoreForMetric(m, i);
                double d2 = e2d_m2.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testEvaluationBinary4dMasking() {
        INDArray prediction = Nd4j.rand(DataType.FLOAT, 2, 3, 10, 10);
        INDArray label = Nd4j.rand(DataType.FLOAT, 2, 3, 10, 10);

        List<INDArray> rowsP = new ArrayList<>();
        List<INDArray> rowsL = new ArrayList<>();

        //Check per-example masking:
        INDArray mask1dPerEx = Nd4j.createFromArray(1, 0);

        NdIndexIterator iter = new NdIndexIterator(2, 10, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            if(mask1dPerEx.getDouble(idx[0]) != 0.0) {
                INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1]), NDArrayIndex.point(idx[2])};
                rowsP.add(prediction.get(idxs));
                rowsL.add(label.get(idxs));
            }
        }

        INDArray p2d = Nd4j.vstack(rowsP);
        INDArray l2d = Nd4j.vstack(rowsL);

        EvaluationBinary e4d_m1 = new EvaluationBinary();
        EvaluationBinary e2d_m1 = new EvaluationBinary();
        e4d_m1.eval(label, prediction, mask1dPerEx);
        e2d_m1.eval(l2d, p2d);
        for(EvaluationBinary.Metric m : EvaluationBinary.Metric.values()){
            for( int i=0; i<3; i++ ) {
                double d1 = e4d_m1.scoreForMetric(m, i);
                double d2 = e2d_m1.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }

        //Check per-output masking:
        INDArray perOutMask = Nd4j.randomBernoulli(0.5, label.shape());
        rowsP.clear();
        rowsL.clear();
        List<INDArray> rowsM = new ArrayList<>();
        iter = new NdIndexIterator(2, 10, 10);
        while (iter.hasNext()) {
            long[] idx = iter.next();
            INDArrayIndex[] idxs = new INDArrayIndex[]{NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[1]), NDArrayIndex.point(idx[2])};
            rowsP.add(prediction.get(idxs));
            rowsL.add(label.get(idxs));
            rowsM.add(perOutMask.get(idxs));
        }
        p2d = Nd4j.vstack(rowsP);
        l2d = Nd4j.vstack(rowsL);
        INDArray m2d = Nd4j.vstack(rowsM);

        EvaluationBinary e3d_m2 = new EvaluationBinary();
        EvaluationBinary e2d_m2 = new EvaluationBinary();
        e3d_m2.eval(label, prediction, perOutMask);
        e2d_m2.eval(l2d, p2d, m2d);
        for(EvaluationBinary.Metric m : EvaluationBinary.Metric.values()){
            for( int i=0; i<3; i++) {
                double d1 = e3d_m2.scoreForMetric(m, i);
                double d2 = e2d_m2.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }
}
