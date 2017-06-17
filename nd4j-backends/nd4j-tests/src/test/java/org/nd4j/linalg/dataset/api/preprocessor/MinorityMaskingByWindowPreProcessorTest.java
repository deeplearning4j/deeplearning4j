package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.classimbalance.MinorityMaskingByWindowPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 6/16/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class MinorityMaskingByWindowPreProcessorTest extends BaseNd4jTest {

    public MinorityMaskingByWindowPreProcessorTest(Nd4jBackend backend) {
        super(backend);
    }

    /*
        Given a dataset
        - with no minority class present, everything should be masked
        - with all minority classes, nothing should be masked
        All checks override for minority/majority class
     */
    @Test
    public void allMajorityOrMinority() {
        float[] someTargets = new float[]{0.01f, 0.1f, 0.5f};
        for (int i = 0; i < someTargets.length; i++) {
            MinorityMaskingByWindowPreProcessor preProcessor = new MinorityMaskingByWindowPreProcessor(someTargets[i], 2);
            //no minority classes present
            DataSet d = makeDataSetSameL(3, 7, new float[]{0.0f, 0.0f, 0.0f}, false);
            DataSet dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            assertEquals(d.getFeatures(), dToPreProcess.getFeatures());
            assertEquals(d.getLabels(), dToPreProcess.getLabels());
            assertEquals(Nd4j.zeros(3, 7), dToPreProcess.getLabelsMaskArray());

            preProcessor.overrideMinorityDefault();
            dToPreProcess = d.copy();
            preProcessor.preProcess(dToPreProcess);
            assertEquals(d.getFeatures(), dToPreProcess.getFeatures());
            assertEquals(d.getLabels(), dToPreProcess.getLabels());
            assertEquals(Nd4j.ones(3, 7), dToPreProcess.getLabelsMaskArray());

            //all minority classes
            MinorityMaskingByWindowPreProcessor preProcessorAllMinority = new MinorityMaskingByWindowPreProcessor(someTargets[i], 2);
            DataSet dAllMinority = makeDataSetSameL(3, 7, new float[]{1.0f, 1.0f, 1.0f}, false);
            dToPreProcess = dAllMinority.copy();
            preProcessorAllMinority.preProcess(dToPreProcess);
            assertEquals(dAllMinority.getFeatures(), dToPreProcess.getFeatures());
            assertEquals(dAllMinority.getLabels(), dToPreProcess.getLabels());
            assertEquals(Nd4j.ones(3, 7), dToPreProcess.getLabelsMaskArray());
        }
    }

    /*
        Different distribution of labels within a minibatch
        Different time series length within a minibatch
        Checks distribution of classes after combining with preprocessed masks
     */
    @Test
    public void mixedDist() {
        int window = 5000;
        int shortSeq = 10000;
        int longSeq = 20000;
        double targetDist = 0.3;
        double tolerancePerc = 0.03; //10% +/- because this is not a very large sample

        //preprocessor should give 30% minority class for every "window"
        MinorityMaskingByWindowPreProcessor preProcessor = new MinorityMaskingByWindowPreProcessor(targetDist, window);

        //construct a dataset with known distribution of minority class and varying time steps
        DataSet batchATimeSteps = makeDataSetSameL(3, shortSeq, new float[]{0.1f, 0.2f, 0.8f}, false);
        DataSet batchBTimeSteps = makeDataSetSameL(3, longSeq, new float[]{0.1f, 0.2f, 0.8f}, false);
        List<DataSet> listofbatches = new ArrayList<>();
        listofbatches.add(batchATimeSteps);
        listofbatches.add(batchBTimeSteps);
        DataSet dataSet = DataSet.merge(listofbatches);
        assertTrue(dataSet.hasMaskArrays());

        //Call preprocess for the same dataset multiple times to mimic calls with .next() and checks total distribution
        int loop = 10;
        for (int i = 0; i < loop; i++) {
            //preprocess dataset
            DataSet dataSetToPreProcess = dataSet.copy();
            preProcessor.preProcess(dataSetToPreProcess);
            INDArray labels = dataSetToPreProcess.getLabels();
            INDArray masks = dataSetToPreProcess.getLabelsMaskArray();

            //check masks are zero where there were no time steps
            INDArray shouldBeAllZeros = masks.get(NDArrayIndex.interval(0, 3), NDArrayIndex.interval(shortSeq, longSeq));
            assertEquals(Nd4j.zeros(shouldBeAllZeros.shape()), shouldBeAllZeros);

            //check distribution of masks in the window length
            for (int j = 0; j < longSeq / window; j++) {
                //collect mask and labels
                INDArray maskWindow = masks.get(NDArrayIndex.all(), NDArrayIndex.interval(j * window, (j + 1) * window));
                INDArray labelWindow = labels.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.interval(j * window, (j + 1) * window)).reshape(6, window);

                //calc minority class distribution after accounting for masks
                INDArray minorityClass = labelWindow.mul(maskWindow);
                INDArray majorityClass = Transforms.not(minorityClass.dup()).mul(maskWindow);
                INDArray minorityDist = minorityClass.sum(1).div(majorityClass.add(minorityClass).sum(1));

                if (j < shortSeq / window) {
                    assertEquals("Failed on window " + j + " batch 0, loop " + i, targetDist, minorityDist.getFloat(0, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 1, loop " + i, targetDist, minorityDist.getFloat(1, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 2, loop " + i, 0.8, minorityDist.getFloat(2, 0), tolerancePerc); //should be unchanged as it was already above target dist
                }
                assertEquals("Failed on window " + j + " batch 3, loop " + i, targetDist, minorityDist.getFloat(3, 0), tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 4, loop " + i, targetDist, minorityDist.getFloat(4, 0), tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 5, loop " + i, 0.8, minorityDist.getFloat(5, 0), tolerancePerc); //should be unchanged as it was already above target dist
            }
        }
    }

    /*
        Same as above but with one hot vectors instead of label size = 1
        Also checks minority override
   */
    @Test
    public void mixedDistOneHot() {
        int window = 5000;
        int shortSeq = 10000;
        int longSeq = 20000;
        double targetDist = 0.3;
        double tolerancePerc = 0.03; //10% +/- because this is not a very large sample

        //preprocessor should give 30% minority class for every "window"
        MinorityMaskingByWindowPreProcessor preProcessor = new MinorityMaskingByWindowPreProcessor(targetDist, window);
        preProcessor.overrideMinorityDefault();

        //construct a dataset with known distribution of minority class and varying time steps
        DataSet batchATimeSteps = makeDataSetSameL(3, shortSeq, new float[]{0.9f, 0.8f, 0.2f}, true);
        DataSet batchBTimeSteps = makeDataSetSameL(3, longSeq, new float[]{0.9f, 0.8f, 0.2f}, true);
        List<DataSet> listofbatches = new ArrayList<>();
        listofbatches.add(batchATimeSteps);
        listofbatches.add(batchBTimeSteps);
        DataSet dataSet = DataSet.merge(listofbatches);
        assertTrue(dataSet.hasMaskArrays());

        //Call preprocess for the same dataset multiple times to mimic calls with .next() and checks total distribution
        int loop = 10;
        for (int i = 0; i < loop; i++) {
            //preprocess dataset
            DataSet dataSetToPreProcess = dataSet.copy();
            preProcessor.preProcess(dataSetToPreProcess);
            INDArray labels = dataSetToPreProcess.getLabels();
            INDArray masks = dataSetToPreProcess.getLabelsMaskArray();

            //check masks are zero where there were no time steps
            INDArray shouldBeAllZeros = masks.get(NDArrayIndex.interval(0, 3), NDArrayIndex.interval(shortSeq, longSeq));
            assertEquals(Nd4j.zeros(shouldBeAllZeros.shape()), shouldBeAllZeros);

            //check distribution of masks in the window length
            for (int j = 0; j < longSeq / window; j++) {
                //collect mask and labels
                INDArray maskWindow = masks.get(NDArrayIndex.all(), NDArrayIndex.interval(j * window, (j + 1) * window));
                INDArray labelWindow = labels.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(j * window, (j + 1) * window));

                //calc minority class distribution after accounting for masks
                INDArray minorityClass = labelWindow.get(NDArrayIndex.all(),NDArrayIndex.point(0),NDArrayIndex.all()).mul(maskWindow);
                INDArray majorityClass = labelWindow.get(NDArrayIndex.all(),NDArrayIndex.point(1),NDArrayIndex.all()).mul(maskWindow);
                //check if labels are actually one hot - they were generated as one hot
                assertEquals(minorityClass.add(majorityClass),Nd4j.ones(6,window).mul(maskWindow));
                INDArray minorityDist = minorityClass.sum(1).div(majorityClass.add(minorityClass).sum(1));

                if (j < shortSeq / window) {
                    assertEquals("Failed on window " + j + " batch 0, loop " + i, targetDist, minorityDist.getFloat(0, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 1, loop " + i, targetDist, minorityDist.getFloat(1, 0), tolerancePerc); //should now be close to target dist
                    assertEquals("Failed on window " + j + " batch 2, loop " + i, 0.8, minorityDist.getFloat(2, 0), tolerancePerc); //should be unchanged as it was already above target dist
                }
                assertEquals("Failed on window " + j + " batch 3, loop " + i, targetDist, minorityDist.getFloat(3, 0), tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 4, loop " + i, targetDist, minorityDist.getFloat(4, 0), tolerancePerc); //should now be close to target dist
                assertEquals("Failed on window " + j + " batch 5, loop " + i, 0.8, minorityDist.getFloat(5, 0), tolerancePerc); //should be unchanged as it was already above target dist
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }


    /*
        Make a random dataset with 0,1 distribution of classes specified
        Will return as a one-hot vector if twoClass = true
     */
    public static DataSet makeDataSetSameL(int batchSize, int timesteps, float[] minorityDist, boolean twoClass) {
        INDArray features = Nd4j.rand(1, batchSize * timesteps * 2).reshape(batchSize, 2, timesteps);
        INDArray labels;
        if (twoClass) {
            labels = Nd4j.zeros(new int[]{batchSize, 2, timesteps});
        } else {
            labels = Nd4j.zeros(new int[]{batchSize, 1, timesteps});
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
