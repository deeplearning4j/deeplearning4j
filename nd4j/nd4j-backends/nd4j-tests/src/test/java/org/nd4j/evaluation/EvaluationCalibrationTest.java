/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.evaluation;

import org.junit.Test;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by Alex on 05/07/2017.
 */
public class EvaluationCalibrationTest extends BaseNd4jTest {

    public EvaluationCalibrationTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testReliabilityDiagram() {

        DataType dtypeBefore = Nd4j.defaultFloatingPointType();
        EvaluationCalibration first = null;
        String sFirst = null;
        try {
            for (DataType globalDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF, DataType.INT}) {
                Nd4j.setDefaultDataTypes(globalDtype, globalDtype.isFPType() ? globalDtype : DataType.DOUBLE);
                for (DataType lpDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {


                    //Test using 5 bins - format: binary softmax-style output
                    //Note: no values fall in fourth bin

                    //[0, 0.2)
                    INDArray bin0Probs = Nd4j.create(new double[][]{{1.0, 0.0}, {0.9, 0.1}, {0.85, 0.15}}).castTo(lpDtype);
                    INDArray bin0Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}).castTo(lpDtype);

                    //[0.2, 0.4)
                    INDArray bin1Probs = Nd4j.create(new double[][]{{0.80, 0.20}, {0.7, 0.3}, {0.65, 0.35}}).castTo(lpDtype);
                    INDArray bin1Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}}).castTo(lpDtype);

                    //[0.4, 0.6)
                    INDArray bin2Probs = Nd4j.create(new double[][]{{0.59, 0.41}, {0.5, 0.5}, {0.45, 0.55}}).castTo(lpDtype);
                    INDArray bin2Labels = Nd4j.create(new double[][]{{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}}).castTo(lpDtype);

                    //[0.6, 0.8)
                    //Empty

                    //[0.8, 1.0]
                    INDArray bin4Probs = Nd4j.create(new double[][]{{0.0, 1.0}, {0.1, 0.9}}).castTo(lpDtype);
                    INDArray bin4Labels = Nd4j.create(new double[][]{{0.0, 1.0}, {0.0, 1.0}}).castTo(lpDtype);


                    INDArray probs = Nd4j.vstack(bin0Probs, bin1Probs, bin2Probs, bin4Probs);
                    INDArray labels = Nd4j.vstack(bin0Labels, bin1Labels, bin2Labels, bin4Labels);

                    EvaluationCalibration ec = new EvaluationCalibration(5, 5);
                    ec.eval(labels, probs);

                    for (int i = 0; i < 1; i++) {
                        double[] avgBinProbsClass;
                        double[] fracPos;
                        if (i == 0) {
                            //Class 0: needs to be handled a little differently, due to threshold/edge cases (0.8, etc)
                            avgBinProbsClass = new double[]{0.05, (0.59 + 0.5 + 0.45) / 3, (0.65 + 0.7) / 2.0,
                                    (0.8 + 0.85 + 0.9 + 1.0) / 4};
                            fracPos = new double[]{0.0 / 2.0, 1.0 / 3, 1.0 / 2, 3.0 / 4};
                        } else {
                            avgBinProbsClass = new double[]{bin0Probs.getColumn(i).meanNumber().doubleValue(),
                                    bin1Probs.getColumn(i).meanNumber().doubleValue(),
                                    bin2Probs.getColumn(i).meanNumber().doubleValue(),
                                    bin4Probs.getColumn(i).meanNumber().doubleValue()};

                            fracPos = new double[]{bin0Labels.getColumn(i).sumNumber().doubleValue() / bin0Labels.size(0),
                                    bin1Labels.getColumn(i).sumNumber().doubleValue() / bin1Labels.size(0),
                                    bin2Labels.getColumn(i).sumNumber().doubleValue() / bin2Labels.size(0),
                                    bin4Labels.getColumn(i).sumNumber().doubleValue() / bin4Labels.size(0)};
                        }

                        org.nd4j.evaluation.curves.ReliabilityDiagram rd = ec.getReliabilityDiagram(i);

                        double[] x = rd.getMeanPredictedValueX();
                        double[] y = rd.getFractionPositivesY();

                        assertArrayEquals(avgBinProbsClass, x, 1e-3);
                        assertArrayEquals(fracPos, y, 1e-3);

                        String s = ec.stats();
                        if(first == null) {
                            first = ec;
                            sFirst = s;
                        } else {
//                            assertEquals(first, ec);
                            assertEquals(sFirst, s);
                            assertTrue(first.getRDiagBinPosCount().equalsWithEps(ec.getRDiagBinPosCount(), lpDtype == DataType.HALF ? 1e-3 : 1e-5));  //Lower precision due to fload
                            assertTrue(first.getRDiagBinTotalCount().equalsWithEps(ec.getRDiagBinTotalCount(), lpDtype == DataType.HALF ? 1e-3 : 1e-5));
                            assertTrue(first.getRDiagBinSumPredictions().equalsWithEps(ec.getRDiagBinSumPredictions(), lpDtype == DataType.HALF ? 1e-3 : 1e-5));
                            assertArrayEquals(first.getLabelCountsEachClass(), ec.getLabelCountsEachClass());
                            assertArrayEquals(first.getPredictionCountsEachClass(), ec.getPredictionCountsEachClass());
                            assertTrue(first.getProbHistogramOverall().equalsWithEps(ec.getProbHistogramOverall(), lpDtype == DataType.HALF ? 1e-3 : 1e-5));
                            assertTrue(first.getProbHistogramByLabelClass().equalsWithEps(ec.getProbHistogramByLabelClass(), lpDtype == DataType.HALF ? 1e-3 : 1e-5));
                        }
                    }
                }
            }
        } finally {
            Nd4j.setDefaultDataTypes(dtypeBefore, dtypeBefore);
        }
    }

    @Test
    public void testLabelAndPredictionCounts() {

        int minibatch = 50;
        int nClasses = 3;

        INDArray arr = Nd4j.rand(minibatch, nClasses);
        arr.diviColumnVector(arr.sum(1));
        INDArray labels = Nd4j.zeros(minibatch, nClasses);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nClasses), 1.0);
        }

        EvaluationCalibration ec = new EvaluationCalibration(5, 5);
        ec.eval(labels, arr);

        int[] expLabelCounts = labels.sum(0).data().asInt();
        int[] expPredictionCount = new int[(int) labels.size(1)];
        INDArray argmax = Nd4j.argMax(arr, 1);
        for (int i = 0; i < argmax.length(); i++) {
            expPredictionCount[argmax.getInt(i)]++;
        }

        assertArrayEquals(expLabelCounts, ec.getLabelCountsEachClass());
        assertArrayEquals(expPredictionCount, ec.getPredictionCountsEachClass());
    }

    @Test
    public void testResidualPlots() {

        int minibatch = 50;
        int nClasses = 3;

        INDArray arr = Nd4j.rand(minibatch, nClasses);
        arr.diviColumnVector(arr.sum(1));
        INDArray labels = Nd4j.zeros(minibatch, nClasses);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nClasses), 1.0);
        }

        int numBins = 5;
        EvaluationCalibration ec = new EvaluationCalibration(numBins, numBins);
        ec.eval(labels, arr);

        INDArray absLabelSubProb = Transforms.abs(labels.sub(arr));
        INDArray argmaxLabels = Nd4j.argMax(labels, 1);

        int[] countsAllClasses = new int[numBins];
        int[][] countsByClass = new int[nClasses][numBins]; //Histogram count of |label[x] - p(x)|; rows x are over classes
        double binSize = 1.0 / numBins;

        for (int i = 0; i < minibatch; i++) {
            int actualClassIdx = argmaxLabels.getInt(i);
            for (int j = 0; j < nClasses; j++) {
                double labelSubProb = absLabelSubProb.getDouble(i, j);
                for (int k = 0; k < numBins; k++) {
                    double binLower = k * binSize;
                    double binUpper = (k + 1) * binSize;
                    if (k == numBins - 1)
                        binUpper = 1.0;

                    if (labelSubProb >= binLower && labelSubProb < binUpper) {
                        countsAllClasses[k]++;
                        if (j == actualClassIdx) {
                            countsByClass[j][k]++;
                        }
                    }
                }
            }
        }

        //Check residual plot - all classes/predictions
        org.nd4j.evaluation.curves.Histogram rpAllClasses = ec.getResidualPlotAllClasses();
        int[] rpAllClassesBinCounts = rpAllClasses.getBinCounts();
        assertArrayEquals(countsAllClasses, rpAllClassesBinCounts);

        //Check residual plot - split by labels for each class
        // i.e., histogram of |label[x] - p(x)| only for those examples where label[x] == 1
        for (int i = 0; i < nClasses; i++) {
            org.nd4j.evaluation.curves.Histogram rpCurrClass = ec.getResidualPlot(i);
            int[] rpCurrClassCounts = rpCurrClass.getBinCounts();

            //            System.out.println(Arrays.toString(countsByClass[i]));
            //            System.out.println(Arrays.toString(rpCurrClassCounts));

            assertArrayEquals("Class: " + i, countsByClass[i], rpCurrClassCounts);
        }



        //Check overall probability distribution
        int[] probCountsAllClasses = new int[numBins];
        int[][] probCountsByClass = new int[nClasses][numBins]; //Histogram count of |label[x] - p(x)|; rows x are over classes
        for (int i = 0; i < minibatch; i++) {
            int actualClassIdx = argmaxLabels.getInt(i);
            for (int j = 0; j < nClasses; j++) {
                double prob = arr.getDouble(i, j);
                for (int k = 0; k < numBins; k++) {
                    double binLower = k * binSize;
                    double binUpper = (k + 1) * binSize;
                    if (k == numBins - 1)
                        binUpper = 1.0;

                    if (prob >= binLower && prob < binUpper) {
                        probCountsAllClasses[k]++;
                        if (j == actualClassIdx) {
                            probCountsByClass[j][k]++;
                        }
                    }
                }
            }
        }

        org.nd4j.evaluation.curves.Histogram allProb = ec.getProbabilityHistogramAllClasses();
        int[] actProbCountsAllClasses = allProb.getBinCounts();

        assertArrayEquals(probCountsAllClasses, actProbCountsAllClasses);

        //Check probability distribution - for each label class
        for (int i = 0; i < nClasses; i++) {
            org.nd4j.evaluation.curves.Histogram probCurrClass = ec.getProbabilityHistogram(i);
            int[] actProbCurrClass = probCurrClass.getBinCounts();

            assertArrayEquals(probCountsByClass[i], actProbCurrClass);
        }
    }

    @Test
    public void testSegmentation(){
        for( int c : new int[]{4, 1}) { //c=1 should be treated as binary classification case
            Nd4j.getRandom().setSeed(12345);
            int mb = 3;
            int h = 3;
            int w = 2;

            //NCHW
            INDArray labels = Nd4j.create(DataType.FLOAT, mb, c, h, w);
            Random r = new Random(12345);
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < h; j++) {
                    for (int k = 0; k < w; k++) {
                        if(c == 1){
                            labels.putScalar(i, 0, j, k, r.nextInt(2));
                        } else {
                            int classIdx = r.nextInt(c);
                            labels.putScalar(i, classIdx, j, k, 1.0);
                        }
                    }
                }
            }

            INDArray predictions = Nd4j.rand(DataType.FLOAT, mb, c, h, w);
            if(c > 1) {
                DynamicCustomOp op = DynamicCustomOp.builder("softmax")
                        .addInputs(predictions)
                        .addOutputs(predictions)
                        .callInplace(true)
                        .addIntegerArguments(1) //Axis
                        .build();
                Nd4j.exec(op);
            }

            EvaluationCalibration e2d = new EvaluationCalibration();
            EvaluationCalibration e4d = new EvaluationCalibration();

            e4d.eval(labels, predictions);

            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < h; j++) {
                    for (int k = 0; k < w; k++) {
                        INDArray rowLabel = labels.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j), NDArrayIndex.point(k));
                        INDArray rowPredictions = predictions.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j), NDArrayIndex.point(k));
                        rowLabel = rowLabel.reshape(1, rowLabel.length());
                        rowPredictions = rowPredictions.reshape(1, rowLabel.length());

                        e2d.eval(rowLabel, rowPredictions);
                    }
                }
            }

            assertEquals(e2d, e4d);


            //NHWC, etc
            INDArray lOrig = labels;
            INDArray fOrig = predictions;
            for (int i = 0; i < 4; i++) {
                switch (i) {
                    case 0:
                        //CNHW - Never really used
                        labels = lOrig.permute(1, 0, 2, 3).dup();
                        predictions = fOrig.permute(1, 0, 2, 3).dup();
                        break;
                    case 1:
                        //NCHW
                        labels = lOrig;
                        predictions = fOrig;
                        break;
                    case 2:
                        //NHCW - Never really used...
                        labels = lOrig.permute(0, 2, 1, 3).dup();
                        predictions = fOrig.permute(0, 2, 1, 3).dup();
                        break;
                    case 3:
                        //NHWC
                        labels = lOrig.permute(0, 2, 3, 1).dup();
                        predictions = fOrig.permute(0, 2, 3, 1).dup();
                        break;
                    default:
                        throw new RuntimeException();
                }

                EvaluationCalibration e = new EvaluationCalibration();
                e.setAxis(i);

                e.eval(labels, predictions);
                assertEquals(e2d, e);
            }
        }
    }

    @Test
    public void testEvaluationCalibration3d() {
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

        EvaluationCalibration e3d = new EvaluationCalibration();
        EvaluationCalibration e2d = new EvaluationCalibration();

        e3d.eval(label, prediction);
        e2d.eval(l2d, p2d);

        System.out.println(e2d.stats());

        assertEquals(e2d, e3d);

        assertEquals(e2d.stats(), e3d.stats());
    }

    @Test
    public void testEvaluationCalibration3dMasking() {
        INDArray prediction = Nd4j.rand(DataType.FLOAT, 2, 3, 10);
        INDArray label = Nd4j.rand(DataType.FLOAT, 2, 3, 10);

        List<INDArray> rowsP = new ArrayList<>();
        List<INDArray> rowsL = new ArrayList<>();

        //Check "DL4J-style" 2d per timestep masking [minibatch, seqLength] mask shape
        INDArray mask2d = Nd4j.randomBernoulli(0.5, 2, 10);
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

        EvaluationCalibration e3d_m2d = new EvaluationCalibration();
        EvaluationCalibration e2d_m2d = new EvaluationCalibration();
        e3d_m2d.eval(label, prediction, mask2d);
        e2d_m2d.eval(l2d, p2d);

        assertEquals(e3d_m2d, e2d_m2d);
    }
}
