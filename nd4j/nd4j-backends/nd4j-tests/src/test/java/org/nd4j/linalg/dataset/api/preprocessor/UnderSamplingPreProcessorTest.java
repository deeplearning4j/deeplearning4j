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

package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.classimbalance.UnderSamplingByMaskingMultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.classimbalance.UnderSamplingByMaskingPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static java.lang.Math.min;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author susaneraly
 */
@Slf4j
@RunWith(Parameterized.class)
public class UnderSamplingPreProcessorTest extends BaseNd4jTest {
    int shortSeq = 10000;
    int longSeq = 20020; //not a perfect multiple of windowSize
    int window = 5000;
    int minibatchSize = 3;
    double targetDist = 0.3;
    double tolerancePerc = 0.03; //10% +/- because this is not a very large sample

    public UnderSamplingPreProcessorTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void allMajority() {
        float[] someTargets = new float[] {0.01f, 0.1f, 0.5f};
        DataSet d = allMajorityDataSet(false);
        DataSet dToPreProcess;
        for (int i = 0; i < someTargets.length; i++) {
            //if all majority default is to mask all time steps
            UnderSamplingByMaskingPreProcessor preProcessor =
                            new UnderSamplingByMaskingPreProcessor(someTargets[i], shortSeq / 2);
            dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            assertEquals(Nd4j.zeros(dToPreProcess.getLabelsMaskArray().shape()), dToPreProcess.getLabelsMaskArray());

            //change default and check distribution which should be 1-targetMinorityDist
            preProcessor.donotMaskAllMajorityWindows();
            dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            INDArray percentagesNow = dToPreProcess.getLabelsMaskArray().sum(1).div(shortSeq);
            assertTrue(Nd4j.valueArrayOf(percentagesNow.shape(), 1 - someTargets[i]).equalsWithEps(percentagesNow,
                            tolerancePerc));
        }
    }

    @Test
    public void allMinority() {
        float[] someTargets = new float[] {0.01f, 0.1f, 0.5f};
        DataSet d = allMinorityDataSet(false);
        DataSet dToPreProcess;
        for (int i = 0; i < someTargets.length; i++) {
            UnderSamplingByMaskingPreProcessor preProcessor =
                            new UnderSamplingByMaskingPreProcessor(someTargets[i], shortSeq / 2);
            dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            //all minority classes present  - check that no time steps are masked
            assertEquals(Nd4j.ones(minibatchSize, shortSeq), dToPreProcess.getLabelsMaskArray());

            //check behavior with override minority - now these are seen as all majority classes
            preProcessor.overrideMinorityDefault();
            preProcessor.donotMaskAllMajorityWindows();
            dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            INDArray percentagesNow = dToPreProcess.getLabelsMaskArray().sum(1).div(shortSeq);
            assertTrue(Nd4j.valueArrayOf(percentagesNow.shape(), 1 - someTargets[i]).equalsWithEps(percentagesNow,
                            tolerancePerc));
        }
    }

    /*
        Different distribution of labels within a minibatch, different time series length within a minibatch
        Checks distribution of classes after preprocessing
     */
    @Test
    public void mixedDist() {

        UnderSamplingByMaskingPreProcessor preProcessor = new UnderSamplingByMaskingPreProcessor(targetDist, window);

        DataSet dataSet = knownDistVariedDataSet(new float[] {0.1f, 0.2f, 0.8f}, false);

        //Call preprocess for the same dataset multiple times to mimic calls with .next() and checks total distribution
        int loop = 2;
        for (int i = 0; i < loop; i++) {
            //preprocess dataset
            DataSet dataSetToPreProcess = dataSet.copy();
            INDArray labelsBefore = dataSetToPreProcess.getLabels().dup();
            preProcessor.preProcess(dataSetToPreProcess);
            INDArray labels = dataSetToPreProcess.getLabels();
            assertEquals(labelsBefore, labels);

            //check masks are zero where there are no time steps
            INDArray masks = dataSetToPreProcess.getLabelsMaskArray();
            INDArray shouldBeAllZeros =
                            masks.get(NDArrayIndex.interval(0, 3), NDArrayIndex.interval(shortSeq, longSeq));
            assertEquals(Nd4j.zeros(shouldBeAllZeros.shape()), shouldBeAllZeros);

            //check distribution of masks in window, going backwards from last time step
            for (int j = (int) Math.ceil((double) longSeq / window); j > 0; j--) {
                //collect mask and labels
                int maxIndex = min(longSeq, j * window);
                int minIndex = min(0, maxIndex - window);
                INDArray maskWindow = masks.get(NDArrayIndex.all(), NDArrayIndex.interval(minIndex, maxIndex));
                INDArray labelWindow = labels.get(NDArrayIndex.all(), NDArrayIndex.point(0),
                                NDArrayIndex.interval(minIndex, maxIndex));

                //calc minority class distribution
                INDArray minorityDist = labelWindow.mul(maskWindow).sum(1).div(maskWindow.sum(1));

                if (j < shortSeq / window) {
                    assertEquals("Failed on window " + j + " batch 0, loop " + i, targetDist,
                                    minorityDist.getFloat(0, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 1, loop " + i, targetDist,
                                    minorityDist.getFloat(1, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 2, loop " + i, 0.8, minorityDist.getFloat(2, 0),
                                    tolerancePerc); //should be unchanged as it was already above target dist
                }
                assertEquals("Failed on window " + j + " batch 3, loop " + i, targetDist, minorityDist.getFloat(3, 0),
                                tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 4, loop " + i, targetDist, minorityDist.getFloat(4, 0),
                                tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 5, loop " + i, 0.8, minorityDist.getFloat(5, 0),
                                tolerancePerc); //should be unchanged as it was already above target dist
            }
        }
    }

    /*
        Same as above but with one hot vectors instead of label size = 1
        Also checks minority override
    */
    @Test
    public void mixedDistOneHot() {

        //preprocessor should give 30% minority class for every "window"
        UnderSamplingByMaskingPreProcessor preProcessor = new UnderSamplingByMaskingPreProcessor(targetDist, window);
        preProcessor.overrideMinorityDefault();

        //construct a dataset with known distribution of minority class and varying time steps
        DataSet dataSet = knownDistVariedDataSet(new float[] {0.9f, 0.8f, 0.2f}, true);

        //Call preprocess for the same dataset multiple times to mimic calls with .next() and checks total distribution
        int loop = 10;
        for (int i = 0; i < loop; i++) {

            //preprocess dataset
            DataSet dataSetToPreProcess = dataSet.copy();
            preProcessor.preProcess(dataSetToPreProcess);
            INDArray labels = dataSetToPreProcess.getLabels();
            INDArray masks = dataSetToPreProcess.getLabelsMaskArray();

            //check masks are zero where there were no time steps
            INDArray shouldBeAllZeros =
                            masks.get(NDArrayIndex.interval(0, 3), NDArrayIndex.interval(shortSeq, longSeq));
            assertEquals(Nd4j.zeros(shouldBeAllZeros.shape()), shouldBeAllZeros);

            //check distribution of masks in the window length, going backwards from last time step
            for (int j = (int) Math.ceil((double) longSeq / window); j > 0; j--) {
                //collect mask and labels
                int maxIndex = min(longSeq, j * window);
                int minIndex = min(0, maxIndex - window);
                INDArray maskWindow = masks.get(NDArrayIndex.all(), NDArrayIndex.interval(minIndex, maxIndex));
                INDArray labelWindow = labels.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(minIndex, maxIndex));

                //calc minority class distribution after accounting for masks
                INDArray minorityClass = labelWindow.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all())
                                .mul(maskWindow);
                INDArray majorityClass = labelWindow.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all())
                                .mul(maskWindow);
                INDArray minorityDist = minorityClass.sum(1).div(majorityClass.add(minorityClass).sum(1));

                if (j < shortSeq / window) {
                    assertEquals("Failed on window " + j + " batch 0, loop " + i, targetDist,
                                    minorityDist.getFloat(0, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 1, loop " + i, targetDist,
                                    minorityDist.getFloat(1, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 2, loop " + i, 0.8, minorityDist.getFloat(2, 0),
                                    tolerancePerc); //should be unchanged as it was already above target dist
                }
                assertEquals("Failed on window " + j + " batch 3, loop " + i, targetDist, minorityDist.getFloat(3, 0),
                                tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 4, loop " + i, targetDist, minorityDist.getFloat(4, 0),
                                tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 5, loop " + i, 0.8, minorityDist.getFloat(5, 0),
                                tolerancePerc); //should be unchanged as it was already above target dist
            }
        }
    }

    //all the tests above into one multidataset
    @Test
    public void testForMultiDataSet() {
        DataSet dataSetA = knownDistVariedDataSet(new float[] {0.8f, 0.1f, 0.2f}, false);
        DataSet dataSetB = knownDistVariedDataSet(new float[] {0.2f, 0.9f, 0.8f}, true);

        HashMap<Integer, Double> targetDists = new HashMap<>();
        targetDists.put(0, 0.5); //balance inputA
        targetDists.put(1, 0.3); //inputB dist = 0.2%
        UnderSamplingByMaskingMultiDataSetPreProcessor maskingMultiDataSetPreProcessor =
                        new UnderSamplingByMaskingMultiDataSetPreProcessor(targetDists, window);
        maskingMultiDataSetPreProcessor.overrideMinorityDefault(1);

        MultiDataSet multiDataSet = fromDataSet(dataSetA, dataSetB);
        maskingMultiDataSetPreProcessor.preProcess(multiDataSet);

        INDArray labels;
        INDArray minorityCount;
        INDArray seqCount;
        INDArray minorityDist;
        //datasetA
        labels = multiDataSet.getLabels(0).reshape(minibatchSize * 2, longSeq).mul(multiDataSet.getLabelsMaskArray(0));
        minorityCount = labels.sum(1);
        seqCount = multiDataSet.getLabelsMaskArray(0).sum(1);
        minorityDist = minorityCount.div(seqCount);
        assertEquals(minorityDist.getDouble(1, 0), 0.5, tolerancePerc);
        assertEquals(minorityDist.getDouble(2, 0), 0.5, tolerancePerc);
        assertEquals(minorityDist.getDouble(4, 0), 0.5, tolerancePerc);
        assertEquals(minorityDist.getDouble(5, 0), 0.5, tolerancePerc);

        //datasetB - override is switched so grab index=0
        labels = multiDataSet.getLabels(1).get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all())
                        .mul(multiDataSet.getLabelsMaskArray(1));
        minorityCount = labels.sum(1);
        seqCount = multiDataSet.getLabelsMaskArray(1).sum(1);
        minorityDist = minorityCount.div(seqCount);
        assertEquals(minorityDist.getDouble(1, 0), 0.3, tolerancePerc);
        assertEquals(minorityDist.getDouble(2, 0), 0.3, tolerancePerc);
        assertEquals(minorityDist.getDouble(4, 0), 0.3, tolerancePerc);
        assertEquals(minorityDist.getDouble(5, 0), 0.3, tolerancePerc);

    }

    @Override
    public char ordering() {
        return 'c';
    }

    public MultiDataSet fromDataSet(DataSet... dataSets) {
        INDArray[] featureArr = new INDArray[dataSets.length];
        INDArray[] labelArr = new INDArray[dataSets.length];
        INDArray[] featureMaskArr = new INDArray[dataSets.length];
        INDArray[] labelMaskArr = new INDArray[dataSets.length];
        for (int i = 0; i < dataSets.length; i++) {
            featureArr[i] = dataSets[i].getFeatures();
            labelArr[i] = dataSets[i].getLabels();
            featureMaskArr[i] = dataSets[i].getFeaturesMaskArray();
            labelMaskArr[i] = dataSets[i].getLabelsMaskArray();
        }
        return new MultiDataSet(featureArr, labelArr, featureMaskArr, labelMaskArr);
    }

    public DataSet allMinorityDataSet(boolean twoClass) {
        return makeDataSetSameL(minibatchSize, shortSeq, new float[] {1.0f, 1.0f, 1.0f}, twoClass);
    }

    public DataSet allMajorityDataSet(boolean twoClass) {
        return makeDataSetSameL(minibatchSize, shortSeq, new float[] {0.0f, 0.0f, 0.0f}, twoClass);
    }

    public DataSet knownDistVariedDataSet(float[] dist, boolean twoClass) {
        //construct a dataset with known distribution of minority class and varying time steps
        DataSet batchATimeSteps = makeDataSetSameL(minibatchSize, shortSeq, dist, twoClass);
        DataSet batchBTimeSteps = makeDataSetSameL(minibatchSize, longSeq, dist, twoClass);
        List<DataSet> listofbatches = new ArrayList<>();
        listofbatches.add(batchATimeSteps);
        listofbatches.add(batchBTimeSteps);
        return DataSet.merge(listofbatches);
    }

    /*
        Make a random dataset with 0,1 distribution of classes specified
        Will return as a one-hot vector if twoClass = true
     */
    public static DataSet makeDataSetSameL(int batchSize, int timesteps, float[] minorityDist, boolean twoClass) {
        INDArray features = Nd4j.rand(1, batchSize * timesteps * 2).reshape(batchSize, 2, timesteps);
        INDArray labels;
        if (twoClass) {
            labels = Nd4j.zeros(new int[] {batchSize, 2, timesteps});
        } else {
            labels = Nd4j.zeros(new int[] {batchSize, 1, timesteps});
        }
        for (int i = 0; i < batchSize; i++) {
            INDArray l;
            if (twoClass) {
                l = labels.get(NDArrayIndex.point(i), NDArrayIndex.point(1), NDArrayIndex.all());
                Nd4j.getExecutioner().exec(new BernoulliDistribution(l, minorityDist[i]));
                INDArray lOther = labels.get(NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.all());
                lOther.assign(Transforms.not(l.dup()));
            } else {
                l = labels.get(NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.all());
                Nd4j.getExecutioner().exec(new BernoulliDistribution(l, minorityDist[i]));
            }
        }
        return new DataSet(features, labels);
    }

}
