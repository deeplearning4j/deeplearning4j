/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@RunWith(Parameterized.class)
public class DataSetTest extends BaseNd4jTest {



    public DataSetTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testViewIterator() {
        DataSetIterator iter = new ViewIterator(new IrisDataSetIterator(150, 150).next(), 10);
        assertTrue(iter.hasNext());
        int count = 0;
        while (iter.hasNext()) {
            DataSet next = iter.next();
            count++;
            assertArrayEquals(new int[] {10, 4}, next.getFeatureMatrix().shape());
        }

        assertFalse(iter.hasNext());
        assertEquals(15, count);
        iter.reset();
        assertTrue(iter.hasNext());
    }

    @Test
    public void  testViewIterator2(){

        INDArray f = Nd4j.linspace(1,100,100).reshape('c', 10, 10);
        DataSet ds = new DataSet(f, f);
        DataSetIterator iter = new ViewIterator(ds, 1);
        for( int i=0; i<10; i++ ){
            assertTrue(iter.hasNext());
            DataSet d = iter.next();
            INDArray exp = f.getRow(i);
            assertEquals(exp, d.getFeatures());
            assertEquals(exp, d.getLabels());
        }
        assertFalse(iter.hasNext());
    }

    @Test
    public void  testViewIterator3(){

        INDArray f = Nd4j.linspace(1,100,100).reshape('c', 10, 10);
        DataSet ds = new DataSet(f, f);
        DataSetIterator iter = new ViewIterator(ds, 6);
        DataSet d1 = iter.next();
        DataSet d2 = iter.next();
        assertFalse(iter.hasNext());
        INDArray e1 = f.get(NDArrayIndex.interval(0,6), NDArrayIndex.all());
        INDArray e2 = f.get(NDArrayIndex.interval(6,10), NDArrayIndex.all());

        assertEquals(e1, d1.getFeatures());
        assertEquals(e2, d2.getFeatures());
    }



    @Test
    public void testSplitTestAndTrain() throws Exception {
        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[] {0, 0, 0, 0, 0, 0, 0, 0}, 1);
        DataSet data = new DataSet(Nd4j.rand(8, 1), labels);

        SplitTestAndTrain train = data.splitTestAndTrain(6, new Random(1));
        assertEquals(train.getTrain().getLabels().length(), 6);

        SplitTestAndTrain train2 = data.splitTestAndTrain(6, new Random(1));
        assertEquals(getFailureMessage(), train.getTrain().getFeatureMatrix(), train2.getTrain().getFeatureMatrix());

        DataSet x0 = new IrisDataSetIterator(150, 150).next();
        SplitTestAndTrain testAndTrain = x0.splitTestAndTrain(10);
        assertArrayEquals(new int[] {10, 4}, testAndTrain.getTrain().getFeatureMatrix().shape());
        assertEquals(x0.getFeatureMatrix().getRows(ArrayUtil.range(0, 10)), testAndTrain.getTrain().getFeatureMatrix());
        assertEquals(x0.getLabels().getRows(ArrayUtil.range(0, 10)), testAndTrain.getTrain().getLabels());


    }

    @Test
    public void testSplitTestAndTrainRng() throws Exception {

        Random rngHere;

        DataSet x1 = new IrisDataSetIterator(150, 150).next(); //original
        DataSet x2 = x1.copy(); //call split test train with rng

        //Manual shuffle
        x1.shuffle(new Random(123).nextLong());
        SplitTestAndTrain testAndTrain = x1.splitTestAndTrain(10);
        // Pass rng with splt test train
        rngHere = new Random(123);
        SplitTestAndTrain testAndTrainRng = x2.splitTestAndTrain(10, rngHere);

        assertArrayEquals(testAndTrainRng.getTrain().getFeatureMatrix().shape(),
                        testAndTrain.getTrain().getFeatureMatrix().shape());
        assertEquals(testAndTrainRng.getTrain().getFeatureMatrix(), testAndTrain.getTrain().getFeatureMatrix());
        assertEquals(testAndTrainRng.getTrain().getLabels(), testAndTrain.getTrain().getLabels());

    }

    @Test
    public void testLabelCounts() {
        DataSet x0 = new IrisDataSetIterator(150, 150).next();
        assertEquals(getFailureMessage(), 0, x0.get(0).outcome());
        assertEquals(getFailureMessage(), 0, x0.get(1).outcome());
        assertEquals(getFailureMessage(), 2, x0.get(149).outcome());
        Map<Integer, Double> counts = x0.labelCounts();
        assertEquals(getFailureMessage(), 50, counts.get(0), 1e-1);
        assertEquals(getFailureMessage(), 50, counts.get(1), 1e-1);
        assertEquals(getFailureMessage(), 50, counts.get(2), 1e-1);

    }

    @Test
    public void testTimeSeriesMerge() {
        //Basic test for time series, all of the same length + no masking arrays
        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int tsLength = 15;

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, tsLength});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, tsLength});
            list.add(new DataSet(in, out));
        }

        DataSet merged = DataSet.merge(list);
        assertEquals(numExamples, merged.numExamples());

        INDArray f = merged.getFeatures();
        INDArray l = merged.getLabels();
        assertArrayEquals(new int[] {numExamples, inSize, tsLength}, f.shape());
        assertArrayEquals(new int[] {numExamples, labelSize, tsLength}, l.shape());

        for (int i = 0; i < numExamples; i++) {
            DataSet exp = list.get(i);
            INDArray expIn = exp.getFeatureMatrix();
            INDArray expL = exp.getLabels();

            INDArray fSubset = f.get(interval(i, i + 1), all(), all());
            INDArray lSubset = l.get(interval(i, i + 1), all(), all());

            assertEquals(expIn, fSubset);
            assertEquals(expL, lSubset);
        }
    }

    @Test
    public void testTimeSeriesMergeDifferentLength() {
        //Test merging of time series with different lengths -> no masking arrays on the input DataSets

        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int minTSLength = 10; //Lengths 10, 11, ..., 19

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});
            list.add(new DataSet(in, out));
        }

        DataSet merged = DataSet.merge(list);
        assertEquals(numExamples, merged.numExamples());

        INDArray f = merged.getFeatures();
        INDArray l = merged.getLabels();
        int expectedLength = minTSLength + numExamples - 1;
        assertArrayEquals(new int[] {numExamples, inSize, expectedLength}, f.shape());
        assertArrayEquals(new int[] {numExamples, labelSize, expectedLength}, l.shape());

        assertTrue(merged.hasMaskArrays());
        assertNotNull(merged.getFeaturesMaskArray());
        assertNotNull(merged.getLabelsMaskArray());
        INDArray featuresMask = merged.getFeaturesMaskArray();
        INDArray labelsMask = merged.getLabelsMaskArray();
        assertArrayEquals(new int[] {numExamples, expectedLength}, featuresMask.shape());
        assertArrayEquals(new int[] {numExamples, expectedLength}, labelsMask.shape());

        //Check each row individually:
        for (int i = 0; i < numExamples; i++) {
            DataSet exp = list.get(i);
            INDArray expIn = exp.getFeatureMatrix();
            INDArray expL = exp.getLabels();

            int thisRowOriginalLength = minTSLength + i;

            INDArray fSubset = f.get(interval(i, i + 1), all(), all());
            INDArray lSubset = l.get(interval(i, i + 1), all(), all());

            for (int j = 0; j < inSize; j++) {
                for (int k = 0; k < thisRowOriginalLength; k++) {
                    double expected = expIn.getDouble(0, j, k);
                    double act = fSubset.getDouble(0, j, k);
                    if (Math.abs(expected - act) > 1e-3) {
                        System.out.println(expIn);
                        System.out.println(fSubset);
                    }
                    assertEquals(expected, act, 1e-3f);
                }

                //Padded values: should be exactly 0.0
                for (int k = thisRowOriginalLength; k < expectedLength; k++) {
                    assertEquals(0.0, fSubset.getDouble(0, j, k), 0.0);
                }
            }

            for (int j = 0; j < labelSize; j++) {
                for (int k = 0; k < thisRowOriginalLength; k++) {
                    double expected = expL.getDouble(0, j, k);
                    double act = lSubset.getDouble(0, j, k);
                    assertEquals(expected, act, 1e-3f);
                }

                //Padded values: should be exactly 0.0
                for (int k = thisRowOriginalLength; k < expectedLength; k++) {
                    assertEquals(0.0, lSubset.getDouble(0, j, k), 0.0);
                }
            }

            //Check mask values:
            for (int j = 0; j < expectedLength; j++) {
                double expected = (j >= thisRowOriginalLength ? 0.0 : 1.0);
                double actFMask = featuresMask.getDouble(i, j);
                double actLMask = labelsMask.getDouble(i, j);

                if (expected != actFMask) {
                    System.out.println(featuresMask);
                    System.out.println(j);
                }

                assertEquals(expected, actFMask, 0.0);
                assertEquals(expected, actLMask, 0.0);
            }
        }
    }


    @Test
    public void testTimeSeriesMergeWithMasking() {
        //Test merging of time series with (a) different lengths, and (b) mask arrays in the input DataSets

        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int minTSLength = 10; //Lengths 10, 11, ..., 19

        Random r = new Random(12345);

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});

            INDArray inMask = Nd4j.create(1, minTSLength + i);
            INDArray outMask = Nd4j.create(1, minTSLength + i);
            for (int j = 0; j < inMask.size(1); j++) {
                inMask.putScalar(j, (r.nextBoolean() ? 1.0 : 0.0));
                outMask.putScalar(j, (r.nextBoolean() ? 1.0 : 0.0));
            }

            list.add(new DataSet(in, out, inMask, outMask));
        }

        DataSet merged = DataSet.merge(list);
        assertEquals(numExamples, merged.numExamples());

        INDArray f = merged.getFeatures();
        INDArray l = merged.getLabels();
        int expectedLength = minTSLength + numExamples - 1;
        assertArrayEquals(new int[] {numExamples, inSize, expectedLength}, f.shape());
        assertArrayEquals(new int[] {numExamples, labelSize, expectedLength}, l.shape());

        assertTrue(merged.hasMaskArrays());
        assertNotNull(merged.getFeaturesMaskArray());
        assertNotNull(merged.getLabelsMaskArray());
        INDArray featuresMask = merged.getFeaturesMaskArray();
        INDArray labelsMask = merged.getLabelsMaskArray();
        assertArrayEquals(new int[] {numExamples, expectedLength}, featuresMask.shape());
        assertArrayEquals(new int[] {numExamples, expectedLength}, labelsMask.shape());

        //Check each row individually:
        for (int i = 0; i < numExamples; i++) {
            DataSet original = list.get(i);
            INDArray expIn = original.getFeatureMatrix();
            INDArray expL = original.getLabels();
            INDArray origMaskF = original.getFeaturesMaskArray();
            INDArray origMaskL = original.getLabelsMaskArray();

            int thisRowOriginalLength = minTSLength + i;

            INDArray fSubset = f.get(interval(i, i + 1), all(), all());
            INDArray lSubset = l.get(interval(i, i + 1), all(), all());

            for (int j = 0; j < inSize; j++) {
                for (int k = 0; k < thisRowOriginalLength; k++) {
                    double expected = expIn.getDouble(0, j, k);
                    double act = fSubset.getDouble(0, j, k);
                    if (Math.abs(expected - act) > 1e-3) {
                        System.out.println(expIn);
                        System.out.println(fSubset);
                    }
                    assertEquals(expected, act, 1e-3f);
                }

                //Padded values: should be exactly 0.0
                for (int k = thisRowOriginalLength; k < expectedLength; k++) {
                    assertEquals(0.0, fSubset.getDouble(0, j, k), 0.0);
                }
            }

            for (int j = 0; j < labelSize; j++) {
                for (int k = 0; k < thisRowOriginalLength; k++) {
                    double expected = expL.getDouble(0, j, k);
                    double act = lSubset.getDouble(0, j, k);
                    assertEquals(expected, act, 1e-3f);
                }

                //Padded values: should be exactly 0.0
                for (int k = thisRowOriginalLength; k < expectedLength; k++) {
                    assertEquals(0.0, lSubset.getDouble(0, j, k), 0.0);
                }
            }

            //Check mask values:
            for (int j = 0; j < expectedLength; j++) {
                double expectedF;
                double expectedL;
                if (j >= thisRowOriginalLength) {
                    //Outside of original data bounds -> should be 0
                    expectedF = 0.0;
                    expectedL = 0.0;
                } else {
                    //Value should be same as original mask array value
                    expectedF = origMaskF.getDouble(j);
                    expectedL = origMaskL.getDouble(j);
                }

                double actFMask = featuresMask.getDouble(i, j);
                double actLMask = labelsMask.getDouble(i, j);
                assertEquals(expectedF, actFMask, 0.0);
                assertEquals(expectedL, actLMask, 0.0);
            }
        }
    }

    @Test
    public void testCnnMerge() {
        //Test merging of CNN data sets
        int nOut = 3;
        int width = 5;
        int height = 4;
        int depth = 3;
        int nExamples1 = 2;
        int nExamples2 = 1;

        int length1 = width * height * depth * nExamples1;
        int length2 = width * height * depth * nExamples2;

        INDArray first = Nd4j.linspace(1, length1, length1).reshape('c', nExamples1, depth, width, height);
        INDArray second = Nd4j.linspace(1, length2, length2).reshape('c', nExamples2, depth, width, height).addi(0.1);

        INDArray labels1 = Nd4j.linspace(1, nExamples1 * nOut, nExamples1 * nOut).reshape('c', nExamples1, nOut);
        INDArray labels2 = Nd4j.linspace(1, nExamples2 * nOut, nExamples2 * nOut).reshape('c', nExamples2, nOut);

        DataSet ds1 = new DataSet(first, labels1);
        DataSet ds2 = new DataSet(second, labels2);

        DataSet merged = DataSet.merge(Arrays.asList(ds1, ds2));

        INDArray fMerged = merged.getFeatureMatrix();
        INDArray lMerged = merged.getLabels();

        assertArrayEquals(new int[] {nExamples1 + nExamples2, depth, width, height}, fMerged.shape());
        assertArrayEquals(new int[] {nExamples1 + nExamples2, nOut}, lMerged.shape());


        assertEquals(first, fMerged.get(interval(0, nExamples1), all(), all(), all()));
        assertEquals(second, fMerged.get(interval(nExamples1, nExamples1 + nExamples2, true), all(), all(), all()));
        assertEquals(labels1, lMerged.get(interval(0, nExamples1), all()));
        assertEquals(labels2, lMerged.get(interval(nExamples1, nExamples1 + nExamples2), all()));


        //Test merging with an empty DataSet (this should be ignored)
        DataSet merged2 = DataSet.merge(Arrays.asList(ds1, new DataSet(), ds2));
        assertEquals(merged, merged2);

        //Test merging with no features in one of the DataSets
        INDArray temp = ds1.getFeatures();
        ds1.setFeatures(null);
        try{
            DataSet.merge(Arrays.asList(ds1, ds2));
            fail("Expected exception");
        } catch (IllegalStateException e){
            //OK
            assertTrue(e.getMessage().contains("Cannot merge"));
        }

        try{
            DataSet.merge(Arrays.asList(ds2, ds1));
            fail("Expected exception");
        } catch (IllegalStateException e){
            //OK
            assertTrue(e.getMessage().contains("Cannot merge"));
        }

        ds1.setFeatures(temp);
        ds2.setLabels(null);
        try{
            DataSet.merge(Arrays.asList(ds1, ds2));
            fail("Expected exception");
        } catch (IllegalStateException e){
            //OK
            assertTrue(e.getMessage().contains("Cannot merge"));
        }

        try{
            DataSet.merge(Arrays.asList(ds2, ds1));
            fail("Expected exception");
        } catch (IllegalStateException e){
            //OK
            assertTrue(e.getMessage().contains("Cannot merge"));
        }
    }

    @Test
    public void testMixedRnn2dMerging() {
        //RNN input with 2d label output
        //Basic test for time series, all of the same length + no masking arrays
        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int tsLength = 15;

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, tsLength});
            INDArray out = Nd4j.rand(new int[] {1, labelSize});
            list.add(new DataSet(in, out));
        }

        DataSet merged = DataSet.merge(list);
        assertEquals(numExamples, merged.numExamples());

        INDArray f = merged.getFeatures();
        INDArray l = merged.getLabels();
        assertArrayEquals(new int[] {numExamples, inSize, tsLength}, f.shape());
        assertArrayEquals(new int[] {numExamples, labelSize}, l.shape());

        for (int i = 0; i < numExamples; i++) {
            DataSet exp = list.get(i);
            INDArray expIn = exp.getFeatureMatrix();
            INDArray expL = exp.getLabels();

            INDArray fSubset = f.get(interval(i, i + 1), all(), all());
            INDArray lSubset = l.get(interval(i, i + 1), all());

            assertEquals(expIn, fSubset);
            assertEquals(expL, lSubset);
        }
    }

    @Test
    public void testMergingWithPerOutputMasking() {

        //Test 2d mask merging, 2d data
        //features
        INDArray f2d1 = Nd4j.create(new double[] {1, 2, 3});
        INDArray f2d2 = Nd4j.create(new double[][] {{4, 5, 6}, {7, 8, 9}});
        //labels
        INDArray l2d1 = Nd4j.create(new double[] {1.5, 2.5, 3.5});
        INDArray l2d2 = Nd4j.create(new double[][] {{4.5, 5.5, 6.5}, {7.5, 8.5, 9.5}});
        //feature masks
        INDArray fm2d1 = Nd4j.create(new double[] {0, 1, 1});
        INDArray fm2d2 = Nd4j.create(new double[][] {{1, 0, 1}, {0, 1, 0}});
        //label masks
        INDArray lm2d1 = Nd4j.create(new double[] {1, 1, 0});
        INDArray lm2d2 = Nd4j.create(new double[][] {{1, 0, 0}, {0, 1, 1}});

        DataSet mds2d1 = new DataSet(f2d1, l2d1, fm2d1, lm2d1);
        DataSet mds2d2 = new DataSet(f2d2, l2d2, fm2d2, lm2d2);
        DataSet merged = DataSet.merge(Arrays.asList(mds2d1, mds2d2));

        INDArray expFeatures2d = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        INDArray expLabels2d = Nd4j.create(new double[][] {{1.5, 2.5, 3.5}, {4.5, 5.5, 6.5}, {7.5, 8.5, 9.5}});
        INDArray expFM2d = Nd4j.create(new double[][] {{0, 1, 1}, {1, 0, 1}, {0, 1, 0}});
        INDArray expLM2d = Nd4j.create(new double[][] {{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});

        DataSet dsExp2d = new DataSet(expFeatures2d, expLabels2d, expFM2d, expLM2d);
        assertEquals(dsExp2d, merged);

        //Test 4d features, 2d labels, 2d masks
        INDArray f4d1 = Nd4j.create(1, 3, 5, 5);
        INDArray f4d2 = Nd4j.create(2, 3, 5, 5);
        DataSet ds4d1 = new DataSet(f4d1, l2d1, null, lm2d1);
        DataSet ds4d2 = new DataSet(f4d2, l2d2, null, lm2d2);
        DataSet merged4d = DataSet.merge(Arrays.asList(ds4d1, ds4d2));
        assertEquals(expLabels2d, merged4d.getLabels());
        assertEquals(expLM2d, merged4d.getLabelsMaskArray());

        //Test 3d mask merging, 3d data
        INDArray f3d1 = Nd4j.create(1, 3, 4);
        INDArray f3d2 = Nd4j.create(1, 3, 3);
        INDArray l3d1 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(1, 3, 4), 0.5));
        INDArray l3d2 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(2, 3, 3), 0.5));
        INDArray lm3d1 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(1, 3, 4), 0.5));
        INDArray lm3d2 = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(2, 3, 3), 0.5));
        DataSet ds3d1 = new DataSet(f3d1, l3d1, null, lm3d1);
        DataSet ds3d2 = new DataSet(f3d2, l3d2, null, lm3d2);

        INDArray expLabels3d = Nd4j.create(3, 3, 4);
        expLabels3d.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)},
                        l3d1);
        expLabels3d.put(new INDArrayIndex[] {NDArrayIndex.interval(1, 2, true), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, 3)}, l3d2);
        INDArray expLM3d = Nd4j.create(3, 3, 4);
        expLM3d.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, 4)},
                        lm3d1);
        expLM3d.put(new INDArrayIndex[] {NDArrayIndex.interval(1, 2, true), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, 3)}, lm3d2);


        DataSet merged3d = DataSet.merge(Arrays.asList(ds3d1, ds3d2));
        assertEquals(expLabels3d, merged3d.getLabels());
        assertEquals(expLM3d, merged3d.getLabelsMaskArray());

        //Test 3d features, 2d masks, 2d output (for example: RNN -> global pooling w/ per-output masking)
        DataSet ds3d2d1 = new DataSet(f3d1, l2d1, null, lm2d1);
        DataSet ds3d2d2 = new DataSet(f3d2, l2d2, null, lm2d2);
        DataSet merged3d2d = DataSet.merge(Arrays.asList(ds3d2d1, ds3d2d2));

        assertEquals(expLabels2d, merged3d2d.getLabels());
        assertEquals(expLM2d, merged3d2d.getLabelsMaskArray());
    }

    @Test
    public void testShuffle4d() {
        int nSamples = 10;
        int nChannels = 3;
        int imgRows = 4;
        int imgCols = 2;

        int nLabels = 5;
        int[] shape = new int[] {nSamples, nChannels, imgRows, imgCols};

        int entries = nSamples * nChannels * imgRows * imgCols;
        int labels = nSamples * nLabels;

        INDArray ds_data = Nd4j.linspace(1, entries, entries).reshape(nSamples, nChannels, imgRows, imgCols);
        INDArray ds_labels = Nd4j.linspace(1, labels, labels).reshape(nSamples, nLabels);
        DataSet ds = new DataSet(ds_data, ds_labels);
        ds.shuffle();

        for (int dim = 1; dim < 4; dim++) {
            //get tensor along dimension - the order in every dimension but zero should be preserved
            for (int tensorNum = 0; tensorNum < entries / shape[dim]; tensorNum++) {
                for (int i = 0, j = 1; j < shape[dim]; i++, j++) {
                    int f_element = ds.getFeatures().tensorAlongDimension(tensorNum, dim).getInt(i);
                    int f_next_element = ds.getFeatures().tensorAlongDimension(tensorNum, dim).getInt(j);
                    int f_element_diff = f_next_element - f_element;
                    assertTrue(f_element_diff == ds_data.stride(dim));
                }
            }
        }
    }

    @Test
    public void testShuffleNd() {
        int numDims = 7;
        int nLabels = 3;
        Random r = new Random();


        int[] shape = new int[numDims];
        int entries = 1;
        for (int i = 0; i < numDims; i++) {
            //randomly generating shapes bigger than 1 
            shape[i] = r.nextInt(4) + 2;
            entries *= shape[i];
        }
        int labels = shape[0] * nLabels;

        INDArray ds_data = Nd4j.linspace(1, entries, entries).reshape(shape);
        INDArray ds_labels = Nd4j.linspace(1, labels, labels).reshape(shape[0], nLabels);

        DataSet ds = new DataSet(ds_data, ds_labels);
        ds.shuffle();

        //Checking Nd dataset which is the data
        for (int dim = 1; dim < numDims; dim++) {
            //get tensor along dimension - the order in every dimension but zero should be preserved
            for (int tensorNum = 0; tensorNum < ds_data.tensorssAlongDimension(dim); tensorNum++) {
                //the difference between consecutive elements should be equal to the stride
                for (int i = 0, j = 1; j < shape[dim]; i++, j++) {
                    int f_element = ds.getFeatures().tensorAlongDimension(tensorNum, dim).getInt(i);
                    int f_next_element = ds.getFeatures().tensorAlongDimension(tensorNum, dim).getInt(j);
                    int f_element_diff = f_next_element - f_element;
                    assertTrue(f_element_diff == ds_data.stride(dim));
                }
            }
        }

        //Checking 2d, features
        int dim = 1;
        //get tensor along dimension - the order in every dimension but zero should be preserved
        for (int tensorNum = 0; tensorNum < ds_labels.tensorssAlongDimension(dim); tensorNum++) {
            //the difference between consecutive elements should be equal to the stride
            for (int i = 0, j = 1; j < nLabels; i++, j++) {
                int l_element = ds.getLabels().tensorAlongDimension(tensorNum, dim).getInt(i);
                int l_next_element = ds.getLabels().tensorAlongDimension(tensorNum, dim).getInt(j);
                int l_element_diff = l_next_element - l_element;
                assertTrue(l_element_diff == ds_labels.stride(dim));
            }
        }
    }

    @Test
    public void testShuffleMeta() {
        int nExamples = 20;
        int nColumns = 4;

        INDArray f = Nd4j.zeros(nExamples, nColumns);
        INDArray l = Nd4j.zeros(nExamples, nColumns);
        List<Integer> meta = new ArrayList<>();

        for (int i = 0; i < nExamples; i++) {
            f.getRow(i).assign(i);
            l.getRow(i).assign(i);
            meta.add(i);
        }

        DataSet ds = new DataSet(f, l);
        ds.setExampleMetaData(meta);

        for (int i = 0; i < 10; i++) {
            ds.shuffle();
            INDArray fCol = f.getColumn(0);
            INDArray lCol = l.getColumn(0);
            System.out.println(fCol + "\t" + ds.getExampleMetaData());
            for (int j = 0; j < nExamples; j++) {
                int fVal = (int) fCol.getDouble(j);
                int lVal = (int) lCol.getDouble(j);
                int metaVal = (Integer) ds.getExampleMetaData().get(j);

                assertEquals(fVal, lVal);
                assertEquals(fVal, metaVal);
            }
        }
    }

    @Test
    public void testLabelNames() {
        List<String> names = Arrays.asList("label1", "label2", "label3", "label0");
        INDArray features = Nd4j.ones(10);
        INDArray labels = Nd4j.linspace(0, 3, 4);
        org.nd4j.linalg.dataset.api.DataSet ds = new DataSet(features, labels);
        ds.setLabelNames(names);
        assertEquals("label1", ds.getLabelName(0));
        assertEquals(4, ds.getLabelNamesList().size());
        assertEquals(names, ds.getLabelNames(labels));
    }

    @Test
    public void testToString() {
        org.nd4j.linalg.dataset.api.DataSet ds = new DataSet();
        //this should not throw a null pointer
        System.out.println(ds);

        //Checking printing of masks
        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int minTSLength = 10; //Lengths 10, 11, ..., 19

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});
            list.add(new DataSet(in, out));
        }

        ds = DataSet.merge(list);
        System.out.println(ds);

    }

    @Test
    public void testGetRangeMask() {
        org.nd4j.linalg.dataset.api.DataSet ds = new DataSet();
        //Checking printing of masks
        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int minTSLength = 10; //Lengths 10, 11, ..., 19

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});
            list.add(new DataSet(in, out));
        }

        int from = 3;
        int to = 9;
        ds = DataSet.merge(list);
        org.nd4j.linalg.dataset.api.DataSet newDs = ds.getRange(from, to);
        //The feature mask does not have to be equal to the label mask, just in this ex it should be
        assertEquals(newDs.getLabelsMaskArray(), newDs.getFeaturesMaskArray());
        //System.out.println(newDs);
        assertEquals(Nd4j.linspace(numExamples + from, numExamples + to - 1, to - from),
                        newDs.getLabelsMaskArray().sum(1));
    }

    @Test
    public void testAsList() {
        org.nd4j.linalg.dataset.api.DataSet ds;
        //Comparing merge with asList
        int numExamples = 10;
        int inSize = 13;
        int labelSize = 5;
        int minTSLength = 10; //Lengths 10, 11, ..., 19

        Nd4j.getRandom().setSeed(12345);
        List<DataSet> list = new ArrayList<>(numExamples);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});
            list.add(new DataSet(in, out));
        }

        //Merged dataset and dataset list
        ds = DataSet.merge(list);
        List<DataSet> dsList = ds.asList();
        //Reset seed
        Nd4j.getRandom().setSeed(12345);
        for (int i = 0; i < numExamples; i++) {
            INDArray in = Nd4j.rand(new int[] {1, inSize, minTSLength + i});
            INDArray out = Nd4j.rand(new int[] {1, labelSize, minTSLength + i});
            DataSet iDataSet = new DataSet(in, out);

            //Checking if the features and labels are equal
            assertEquals(iDataSet.getFeatures(),
                            dsList.get(i).getFeatures().get(all(), all(), interval(0, minTSLength + i)));
            assertEquals(iDataSet.getLabels(),
                            dsList.get(i).getLabels().get(all(), all(), interval(0, minTSLength + i)));
        }
    }


    @Test
    public void testDataSetSaveLoad() throws IOException {

        boolean[] b = new boolean[] {true, false};

        INDArray f = Nd4j.linspace(1, 24, 24).reshape('c', 4, 3, 2);
        INDArray l = Nd4j.linspace(24, 48, 24).reshape('c', 4, 3, 2);
        INDArray fm = Nd4j.linspace(100, 108, 8).reshape('c', 4, 2);
        INDArray lm = Nd4j.linspace(108, 116, 8).reshape('c', 4, 2);

        for (boolean features : b) {
            for (boolean labels : b) {
                for (boolean labelsSameAsFeatures : b) {
                    if (labelsSameAsFeatures && (!features || !labels))
                        continue; //Can't have "labels same as features" if no features, or if no labels

                    for (boolean fMask : b) {
                        for (boolean lMask : b) {

                            DataSet ds = new DataSet((features ? f : null),
                                            (labels ? (labelsSameAsFeatures ? f : l) : null), (fMask ? fm : null),
                                            (lMask ? lm : null));

                            ByteArrayOutputStream baos = new ByteArrayOutputStream();
                            DataOutputStream dos = new DataOutputStream(baos);

                            ds.save(dos);

                            byte[] asBytes = baos.toByteArray();

                            ByteArrayInputStream bais = new ByteArrayInputStream(asBytes);
                            DataInputStream dis = new DataInputStream(bais);

                            DataSet ds2 = new DataSet();
                            ds2.load(dis);
                            dis.close();

                            assertEquals(ds, ds2);

                            if (labelsSameAsFeatures)
                                assertTrue(ds2.getFeatureMatrix() == ds2.getLabels()); //Expect same object
                        }
                    }
                }
            }
        }
    }


    @Test
    public void testDataSetSaveLoadSingle() throws IOException {

        INDArray f = Nd4j.linspace(1, 24, 24).reshape('c', 4, 3, 2);
        INDArray l = Nd4j.linspace(24, 48, 24).reshape('c', 4, 3, 2);
        INDArray fm = Nd4j.linspace(100, 108, 8).reshape('c', 4, 2);
        INDArray lm = Nd4j.linspace(108, 116, 8).reshape('c', 4, 2);

        boolean features = true;
        boolean labels = false;
        boolean labelsSameAsFeatures = false;
        boolean fMask = true;
        boolean lMask = true;

        DataSet ds = new DataSet((features ? f : null), (labels ? (labelsSameAsFeatures ? f : l) : null),
                        (fMask ? fm : null), (lMask ? lm : null));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);

        ds.save(dos);
        dos.close();

        byte[] asBytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(asBytes);
        DataInputStream dis = new DataInputStream(bais);

        DataSet ds2 = new DataSet();
        ds2.load(dis);
        dis.close();

        assertEquals(ds, ds2);

        if (labelsSameAsFeatures)
            assertTrue(ds2.getFeatureMatrix() == ds2.getLabels()); //Expect same object
    }

    @Test
    public void testMdsShuffle(){

        MultiDataSet orig = new MultiDataSet(Nd4j.linspace(1,100,100).reshape('c',10,10),
                Nd4j.linspace(100,200,100).reshape('c',10,10));

        MultiDataSet mds = new MultiDataSet(Nd4j.linspace(1,100,100).reshape('c',10,10),
                Nd4j.linspace(100,200,100).reshape('c',10,10));
        mds.shuffle();

        assertNotEquals(orig, mds);

        boolean[] foundF = new boolean[10];
        boolean[] foundL = new boolean[10];

        for( int i=0; i<10; i++ ){
            double f = mds.getFeatures(0).getDouble(i,0);
            double l = mds.getLabels(0).getDouble(i,0);

            int fi = (int)(f/10.0);   //21.0 -> 2, etc
            int li = (int)((l-100)/10.0);   //121.0 -> 2

            foundF[fi] = true;
            foundL[li] = true;
        }

        boolean allF = true;
        boolean allL = true;
        for( int i=0; i<10; i++ ){
            allF &= foundF[i];
            allL &= foundL[i];
        }

        assertTrue(allF);
        assertTrue(allL);
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
