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
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.evaluation.curves.PrecisionRecallCurve;
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

/**
 * Created by Alex on 21/03/2017.
 */
public class ROCBinaryTest extends BaseNd4jTest {

    public ROCBinaryTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testROCBinary() {
        //Compare ROCBinary to ROC class

        DataType dtypeBefore = Nd4j.defaultFloatingPointType();
        ROCBinary first30 = null;
        ROCBinary first0 = null;
        String sFirst30 = null;
        String sFirst0 = null;
        try {
            for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT}) {
//            for (DataType globalDtype : new DataType[]{DataType.HALF}) {
                Nd4j.setDefaultDataTypes(globalDtype, globalDtype.isFPType() ? globalDtype : DataType.DOUBLE);
                for (DataType lpDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
                    String msg = "globalDtype=" + globalDtype + ", labelPredictionsDtype=" + lpDtype;

                    int nExamples = 50;
                    int nOut = 4;
                    long[] shape = {nExamples, nOut};

                    for (int thresholdSteps : new int[]{30, 0}) { //0 == exact

                        Nd4j.getRandom().setSeed(12345);
                        INDArray labels =
                                Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(DataType.DOUBLE, shape), 0.5)).castTo(lpDtype);

                        Nd4j.getRandom().setSeed(12345);
                        INDArray predicted = Nd4j.rand(DataType.DOUBLE, shape).castTo(lpDtype);

                        ROCBinary rb = new ROCBinary(thresholdSteps);

                        for (int xe = 0; xe < 2; xe++) {
                            rb.eval(labels, predicted);

                            //System.out.println(rb.stats());

                            double eps = lpDtype == DataType.HALF ? 1e-2 : 1e-6;
                            for (int i = 0; i < nOut; i++) {
                                INDArray lCol = labels.getColumn(i, true);
                                INDArray pCol = predicted.getColumn(i, true);


                                ROC r = new ROC(thresholdSteps);
                                r.eval(lCol, pCol);

                                double aucExp = r.calculateAUC();
                                double auc = rb.calculateAUC(i);

                                assertEquals(msg, aucExp, auc, eps);

                                long apExp = r.getCountActualPositive();
                                long ap = rb.getCountActualPositive(i);
                                assertEquals(msg, ap, apExp);

                                long anExp = r.getCountActualNegative();
                                long an = rb.getCountActualNegative(i);
                                assertEquals(anExp, an);

                                PrecisionRecallCurve pExp = r.getPrecisionRecallCurve();
                                PrecisionRecallCurve p = rb.getPrecisionRecallCurve(i);

                                assertEquals(msg, pExp, p);
                            }

                            String s = rb.stats();

                            if(thresholdSteps == 0){
                                if(first0 == null) {
                                    first0 = rb;
                                    sFirst0 = s;
                                } else if(lpDtype != DataType.HALF) {   //Precision issues with FP16
                                    assertEquals(msg, sFirst0, s);
                                    assertEquals(first0, rb);
                                }
                            } else {
                                if(first30 == null) {
                                    first30 = rb;
                                    sFirst30 = s;
                                } else if(lpDtype != DataType.HALF) {   //Precision issues with FP16
                                    assertEquals(msg, sFirst30, s);
                                    assertEquals(first30, rb);
                                }
                            }

//                            rb.reset();
                            rb = new ROCBinary(thresholdSteps);
                        }
                    }
                }
            }
        } finally {
            Nd4j.setDefaultDataTypes(dtypeBefore, dtypeBefore);
        }
    }

    @Test
    public void testRocBinaryMerging() {
        for (int nSteps : new int[]{30, 0}) { //0 == exact
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
    }


    @Test
    public void testROCBinaryPerOutputMasking() {

        for (int nSteps : new int[]{30, 0}) { //0 == exact

            //Here: we'll create a test array, then insert some 'masked out' values, and ensure we get the same results
            INDArray mask = Nd4j.create(new double[][]{{1, 1, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}});

            INDArray labels = Nd4j.create(new double[][]{{0, 1, 0}, {1, 1, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1}});

            //Remove the 1 masked value for each column
            INDArray labelsExMasked = Nd4j.create(new double[][]{{0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}});

            INDArray predicted = Nd4j.create(new double[][]{{0.9, 0.4, 0.6}, {0.2, 0.8, 0.4}, {0.6, 0.1, 0.1},
                    {0.3, 0.7, 0.2}, {0.8, 0.6, 0.6}});

            INDArray predictedExMasked = Nd4j.create(
                    new double[][]{{0.9, 0.4, 0.6}, {0.6, 0.8, 0.4}, {0.3, 0.7, 0.1}, {0.8, 0.6, 0.6}});

            ROCBinary rbMasked = new ROCBinary(nSteps);
            rbMasked.eval(labels, predicted, mask);

            ROCBinary rb = new ROCBinary(nSteps);
            rb.eval(labelsExMasked, predictedExMasked);

            String s1 = rb.stats();
            String s2 = rbMasked.stats();
            assertEquals(s1, s2);

            for (int i = 0; i < 3; i++) {
                PrecisionRecallCurve pExp = rb.getPrecisionRecallCurve(i);
                PrecisionRecallCurve p = rbMasked.getPrecisionRecallCurve(i);

                assertEquals(pExp, p);
            }
        }
    }



    @Test
    public void testROCBinary3d() {
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

        ROCBinary e3d = new ROCBinary();
        ROCBinary e2d = new ROCBinary();

        e3d.eval(label, prediction);
        e2d.eval(l2d, p2d);

        for (ROCBinary.Metric m : ROCBinary.Metric.values()) {
            for( int i=0; i<5; i++ ) {
                double d1 = e3d.scoreForMetric(m, i);
                double d2 = e2d.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testROCBinary4d() {
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

        ROCBinary e4d = new ROCBinary();
        ROCBinary e2d = new ROCBinary();

        e4d.eval(label, prediction);
        e2d.eval(l2d, p2d);

        for (ROCBinary.Metric m : ROCBinary.Metric.values()) {
            for( int i=0; i<3; i++ ) {
                double d1 = e4d.scoreForMetric(m, i);
                double d2 = e2d.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testROCBinary3dMasking() {
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

        ROCBinary e3d_m2d = new ROCBinary();
        ROCBinary e2d_m2d = new ROCBinary();
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

        ROCBinary e4d_m2 = new ROCBinary();
        ROCBinary e2d_m2 = new ROCBinary();
        e4d_m2.eval(label, prediction, perOutMask);
        e2d_m2.eval(l2d, p2d, m2d);
        for(ROCBinary.Metric m : ROCBinary.Metric.values()){
            for(int i=0; i<3; i++ ) {
                double d1 = e4d_m2.scoreForMetric(m, i);
                double d2 = e2d_m2.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }

    @Test
    public void testROCBinary4dMasking() {
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

        ROCBinary e4d_m1 = new ROCBinary();
        ROCBinary e2d_m1 = new ROCBinary();
        e4d_m1.eval(label, prediction, mask1dPerEx);
        e2d_m1.eval(l2d, p2d);
        for(ROCBinary.Metric m : ROCBinary.Metric.values()){
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

        ROCBinary e3d_m2 = new ROCBinary();
        ROCBinary e2d_m2 = new ROCBinary();
        e3d_m2.eval(label, prediction, perOutMask);
        e2d_m2.eval(l2d, p2d, m2d);
        for(ROCBinary.Metric m : ROCBinary.Metric.values()){
            for( int i=0; i<3; i++) {
                double d1 = e3d_m2.scoreForMetric(m, i);
                double d2 = e2d_m2.scoreForMetric(m, i);
                assertEquals(m.toString(), d2, d1, 1e-6);
            }
        }
    }
}
