package org.nd4j.linalg.dataset;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by susaneraly on 7/15/16.
 */
@RunWith(Parameterized.class)
public class PreProcessor3D4DTest extends BaseNd4jTest {

    public PreProcessor3D4DTest(Nd4jBackend backend) {
        super(backend);
    }
    @Test
    public void testBruteForce3d() {
        /*
           This is 3d dataset where dimensions are sample#,features,timesteps
           Timesteps are set to consecutive nums continuing across samples
           Each feature is a multiple of the other. Yes they are collinear but this is a test :)
           The obtained values are compared to the theoretical mean and std dev
         */
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();

        int timeSteps = 12;
        int samples = 10;
        //multiplier for the features
        INDArray featureScaleA = Nd4j.create(new double [] {1,-2,3}).reshape(3,1);
        INDArray featureScaleB = Nd4j.create(new double [] {2,2,3}).reshape(3,1);

        Construct3dDataSet caseA = new Construct3dDataSet(featureScaleA,timeSteps,samples,1);
        Construct3dDataSet caseB = new Construct3dDataSet(featureScaleB,timeSteps,samples,1);

        myNormalizer.fit(caseA.sampleDataSet);
        assertEquals(caseA.expectedMean.sub(myNormalizer.getMean()));
        assertEquals(caseA.expectedStd,myNormalizer.getStd());

        myMinMaxScaler.fit(caseB.sampleDataSet);
        assertEquals(caseB.expectedMin,myMinMaxScaler.getMin());
        assertEquals(caseB.expectedMax,myMinMaxScaler.getMax());

    }


    @Test
    public void testBruteForce3dMask() {
        /*
           This is 3d dataset where dimensions are sample#,features,timesteps
           Timesteps are set to consecutive nums continuing across samples
           Each feature is a multiple of the other. Yes they are collinear but this is a test :)
           The obtained values are compared to the theoretical mean and std dev
         */

        //Generating data for test
        int samples = 100;

        INDArray featureScale = Nd4j.create(new float[] {1,2,10});
        int timeStepsU = 5;
        Construct3dDataSet sampleU = new Construct3dDataSet(featureScale,timeStepsU,samples,1);

        int timeStepsV = 3;
        Construct3dDataSet sampleV = new Construct3dDataSet(featureScale,timeStepsV,samples,sampleU.newOrigin);
        //int totalTimeSteps = timeSteps;

        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(sampleU.sampleDataSet);
        dataSetList.add(sampleV.sampleDataSet);
        DataSet fullDataSetUV = DataSet.merge(dataSetList);
        DataSet fullDataSetUVAnother = fullDataSetUV.copy();

        /*
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        //checking standardize first
        myNormalizer.fit(fullDataSetUV);
        myMinMaxScaler.fit(fullDataSetUVAnother);
        // The theoretical mean should be the mean of 1,..samples*timesteps
        int maxN = samples*totalTimeSteps;
        float theoreticalMean = (maxN + 1)/2.0f;
        INDArray expectedMean = Nd4j.create(new float[] {theoreticalMean,theoreticalMean,theoreticalMean}).reshape(3,1);
        expectedMean.muliColumnVector(featureScale);

        float stdNaturalNums = (float) Math.sqrt((maxN*maxN- 1)/12);
        INDArray expectedStd = Nd4j.create(new float[] {stdNaturalNums,stdNaturalNums,stdNaturalNums}).reshape(3,1);
        expectedStd.muliColumnVector(featureScale);
        //std calculates the sample std so divides by (n-1) not n
        expectedStd.muli(Math.sqrt(maxN)).divi(Math.sqrt(maxN-1));

        INDArray theoreticalMin = featureScale.transpose();
        INDArray theoreticalMax = Nd4j.ones(theoreticalMin.shape()).muli(maxN).muli(featureScale);

        assertEquals(myNormalizer.getMean(),expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        assertEquals(myMinMaxScaler.getMin(),theoreticalMin);
        assertEquals(myMinMaxScaler.getMax(),theoreticalMax);

        //Same Test with an Iterator
        DataSetIterator sampleIter = new TestDataSetIterator(fullDataSetUV,5);
        DataSetIterator sampleIterAnother = new TestDataSetIterator(fullDataSetUVAnother,5);
        myNormalizer.fit(sampleIter);
        myMinMaxScaler.fit(sampleIterAnother);
        assertEquals(myNormalizer.getMean(),expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        assertEquals(myMinMaxScaler.getMin(),theoreticalMin);
        assertEquals(myMinMaxScaler.getMax(),theoreticalMax);
        */

    }

    @Test
    public void testBruteForce3dMaskLabels() {
        /*
           This is 3d dataset where dimensions are sample#,features,timesteps
           Timesteps are set to consecutive nums continuing across samples
           Each feature is a multiple of the other. Yes they are collinear but this is a test :)
           The obtained values are compared to the theoretical mean and std dev
         */

        /*
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fitLabel(true);

        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fitLabel(true);

        int timeSteps = 5;
        int totalTimeSteps = timeSteps;
        int samples = 100;
        //multiplier for the features
        INDArray featureScale = Nd4j.create(3,1);
        featureScale.put(0,0,1);
        featureScale.put(1,0,2);
        featureScale.put(2,0,10);

        INDArray template = Nd4j.linspace(1,timeSteps,timeSteps);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template.muliColumnVector(featureScale);
        template = template.reshape(1,3,timeSteps);
        INDArray featureMatrix = template.dup();
        // /\ /\ i = 0
        //the rest \/\/
        int newStart = timeSteps+1;
        int newEnd;
        for (int i=1;i<samples;i++) {
            newEnd = newStart + timeSteps - 1;
            template = Nd4j.linspace(newStart,newEnd,timeSteps);
            template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
            template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
            template.muliColumnVector(featureScale);
            template = template.reshape(1,3,timeSteps);
            newStart = newEnd + 1;
            featureMatrix = Nd4j.concat(0,featureMatrix,template);
        }
        INDArray labelSet = featureMatrix.dup();
        DataSet sampleDataSetU = new DataSet(featureMatrix, labelSet);


        //Continuing on with a new dataset to merge with the one above
        timeSteps = 3;
        totalTimeSteps += timeSteps;
        newEnd = newStart + timeSteps - 1;
        template = Nd4j.linspace(newStart,newEnd,timeSteps);
        template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
        template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
        template.muliColumnVector(featureScale);
        template = template.reshape(1,3,timeSteps);
        featureMatrix = template.dup();
        newStart = newEnd + 1;
        // /\ /\ i = 0
        //the rest \/\/
        for (int i=1;i<samples;i++) {
            newEnd = newStart + timeSteps - 1;
            template = Nd4j.linspace(newStart,newEnd,timeSteps);
            template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
            template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
            template.muliColumnVector(featureScale);
            template = template.reshape(1,3,timeSteps);
            newStart = newEnd + 1;
            featureMatrix = Nd4j.concat(0,featureMatrix,template);
        }
        labelSet = featureMatrix.dup();
        DataSet sampleDataSetV = new DataSet(featureMatrix, labelSet);


        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(sampleDataSetU);
        dataSetList.add(sampleDataSetV);
        DataSet fullDataSetUV = DataSet.merge(dataSetList);
        DataSet fullDataSetUVAnother = fullDataSetUV.copy();

        myNormalizer.fit(fullDataSetUV);
        myMinMaxScaler.fit(fullDataSetUVAnother);
        // The theoretical mean should be the mean of 1,..samples*timesteps
        int maxN = samples*totalTimeSteps;
        float theoreticalMean = (maxN + 1)/2.0f;
        INDArray expectedMean = Nd4j.create(new float[] {theoreticalMean,theoreticalMean,theoreticalMean}).reshape(3,1);
        expectedMean.muliColumnVector(featureScale);

        float stdNaturalNums = (float) Math.sqrt((maxN*maxN- 1)/12);
        INDArray expectedStd = Nd4j.create(new float[] {stdNaturalNums,stdNaturalNums,stdNaturalNums}).reshape(3,1);
        expectedStd.muliColumnVector(featureScale);
        //std calculates the sample std so divides by (n-1) not n
        expectedStd.muli(Math.sqrt(maxN)).divi(Math.sqrt(maxN-1));

        assertEquals(myNormalizer.getMean(),expectedMean);
        assertEquals(myNormalizer.getLabelMean(),expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        assertTrue(Transforms.abs(myNormalizer.getLabelStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);

        INDArray theoreticalMin = featureScale.transpose();
        INDArray theoreticalMax = Nd4j.ones(theoreticalMin.shape()).muli(maxN).muli(featureScale);
        assertEquals(myMinMaxScaler.getMin(),theoreticalMin);
        assertEquals(myMinMaxScaler.getMax(),theoreticalMax);

        //Same Test with an Iterator
        DataSetIterator sampleIter = new TestDataSetIterator(fullDataSetUV,5);
        myNormalizer.fit(sampleIter);
        System.out.println("Testing with an iterator...");
        System.out.println("Testing mean with an iterator...");
        assertEquals(myNormalizer.getMean(),expectedMean);
        assertEquals(myNormalizer.getLabelMean(),expectedMean);
        System.out.println("Testing std with an iterator...");
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        assertTrue(Transforms.abs(myNormalizer.getLabelStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);

        DataSetIterator sampleIterAnother = new TestDataSetIterator(fullDataSetUVAnother,5);
        myMinMaxScaler.fit(sampleIterAnother);
        assertEquals(myMinMaxScaler.getMin(),theoreticalMin);
        assertEquals(myMinMaxScaler.getLabelMin(),theoreticalMin);
        assertEquals(myMinMaxScaler.getMax(),theoreticalMax);
        assertEquals(myMinMaxScaler.getLabelMax(),theoreticalMax);
        */

    }

    @Test
    public void testBruteForce4d() {
        //this is an image - #of images x channels x size x size
        // test with 2 samples, 3 channels x 10 x 10
        // three channels are in the ratio 1:2:3
        // the two samples are multiples of each other 1:3
        /*
        INDArray oneChannel = Nd4j.linspace(1,100,100).reshape(1,10,10);
        INDArray imageOne = Nd4j.concat(0,oneChannel,oneChannel.mul(2),oneChannel.mul(3)).reshape(1,3,10,10);
        INDArray imageTwo = imageOne.mul(3);

        INDArray allImages = Nd4j.concat(0,imageOne,imageTwo);
        INDArray labels = Nd4j.linspace(50,100,2).reshape(2,1);

        DataSet dataSet = new DataSet(allImages,labels);
        DataSet dataSetAnother = dataSet.copy();

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myNormalizer.fitLabel(true);
        myNormalizer.fit(dataSet);
        //works out to be 1->100 and 3*(1->100)
        //mean is (101/2 + 3*101/2)/2 = 101
        // std is 82.1599
        INDArray expectedMean = Nd4j.linspace(1,3,3).reshape(1,3).mul(101);
        INDArray expectedStd = Nd4j.linspace(1,3,3).reshape(1,3).mul(82.1599);
        INDArray expectedLabelMean = labels.mean(0);
        INDArray expectedLabelStd = labels.std(0);
        assertTrue(Transforms.abs(expectedMean.sub(myNormalizer.getMean()).div(expectedMean)).maxNumber().floatValue() < 0.03f);
        assertTrue(Transforms.abs(expectedLabelMean.sub(myNormalizer.getLabelMean()).div(expectedLabelMean)).maxNumber().floatValue() < 0.03f);
        assertTrue(Transforms.abs(expectedStd.sub(myNormalizer.getStd()).div(expectedStd)).maxNumber().floatValue() < 0.03f);
        assertTrue(Transforms.abs(expectedLabelStd.sub(myNormalizer.getLabelStd()).div(expectedLabelStd)).maxNumber().floatValue() < 0.03f);

        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fitLabel(true);
        myMinMaxScaler.fit(dataSetAnother);
        //channel 1: min-1,max-100*3
        //channel 2: min-2,max-200*3
        //channel 1: min-3,max-300*3
        INDArray expectedMin = Nd4j.create(new double[] {1,2,3});
        INDArray expectedMax = Nd4j.create(new double[] {300,600,900});
        assertEquals(expectedMin,myMinMaxScaler.getMin());
        assertEquals(expectedMax,myMinMaxScaler.getMax());

        DataSet copyDataSet = dataSet.copy();
        myNormalizer.transform(copyDataSet);
        //all the channels should have the same value now -> since they are multiples of each other
        //across images (x-k1)/k2 and (3x-k1)/k2
        //difference is 3x/k2 - k1/k2 - x/k2 + k1/k2 = 2x/k2; k2 is the stdDev

        //checks to see if all values are the same
        INDArray transformedVals = copyDataSet.getFeatures();
        INDArray imageUno = transformedVals.slice(0);
        assertEquals(imageUno.slice(0),imageUno.slice(1));
        assertEquals(imageUno.slice(0),imageUno.slice(2));

        INDArray imageDos = transformedVals.slice(1);
        INDArray diffUnoDos = imageDos.sub(imageUno);
        INDArray divIs = dataSet.getFeatures().slice(0).div(diffUnoDos).mul(2); //should be the std dev now
        //System.out.println(divIs);

        INDArray template = Nd4j.ones(new int[] {1,10,10});
        INDArray expecteddiv = Nd4j.concat(0,
                template.mul(myNormalizer.getStd().getDouble(0,0)),
                template.mul(myNormalizer.getStd().getDouble(0,1)),
                template.mul(myNormalizer.getStd().getDouble(0,2))
                );
        assertTrue(Transforms.abs(expecteddiv.sub(divIs)).maxNumber().floatValue() < 0.001);
        assertEquals(labels.sub(expectedLabelMean).div(expectedLabelStd),copyDataSet.getLabels());
        */
    }

    public class Construct3dDataSet {

         /*
            This will return a dataset where the features are consecutive numbers scaled by featureScaler (a column vector)
            If more than one sample is specified it will continue the series from the last sample
            If origin is not 1, the series will start from the value given
             */
        DataSet sampleDataSet;
        INDArray featureScale;
        int numFeatures,maxN,timeSteps,samples,origin,newOrigin;
        INDArray expectedMean, expectedStd, expectedMin, expectedMax;

        public Construct3dDataSet(INDArray featureScale, int timeSteps, int samples, int origin) {
            this.featureScale = featureScale;
            this.timeSteps = timeSteps;
            this.samples = samples;
            this.origin = origin;
            numFeatures = featureScale.size(0);
            maxN = samples * timeSteps;
            INDArray template = Nd4j.linspace(origin, origin+timeSteps-1, timeSteps);
            template = Nd4j.concat(0, Nd4j.linspace(origin, origin+timeSteps-1, timeSteps), template);
            template = Nd4j.concat(0, Nd4j.linspace(origin, origin+timeSteps-1, timeSteps), template);
            template.muliColumnVector(featureScale);
            template = template.reshape(1, numFeatures, timeSteps);
            INDArray featureMatrix = template.dup();

            int newStart = origin+timeSteps;
            int newEnd;
            for (int i = 1; i < samples; i++) {
                newEnd = newStart + timeSteps - 1;
                template = Nd4j.linspace(newStart, newEnd, timeSteps);
                template = Nd4j.concat(0, Nd4j.linspace(newStart, newEnd, timeSteps), template);
                template = Nd4j.concat(0, Nd4j.linspace(newStart, newEnd, timeSteps), template);
                template.muliColumnVector(featureScale);
                template = template.reshape(1, numFeatures, timeSteps);
                newStart = newEnd + 1;
                featureMatrix = Nd4j.concat(0, featureMatrix, template);
            }
            INDArray labelSet = featureMatrix.dup();
            this.newOrigin = newStart;
            sampleDataSet = new DataSet(featureMatrix, labelSet);

            //calculating stats
            // The theoretical mean should be the mean of 1,..samples*timesteps
            float theoreticalMean = origin - 1 + (samples*timeSteps + 1)/2.0f;
            expectedMean = Nd4j.create(new double[] {theoreticalMean,theoreticalMean,theoreticalMean}).reshape(3,1);
            expectedMean.muliColumnVector(featureScale);

            float stdNaturalNums = (float) Math.sqrt((samples*samples*timeSteps*timeSteps - 1)/12);
            expectedStd = Nd4j.create(new float[] {stdNaturalNums,stdNaturalNums,stdNaturalNums}).reshape(3,1);
            expectedStd.muliColumnVector(Transforms.abs(featureScale,true));
            //std calculates the sample std so divides by (n-1) not n
            expectedStd.muli(Math.sqrt(maxN)).divi(Math.sqrt(maxN-1));

            //min max assumes all scaling values are +ve
            expectedMin = Nd4j.ones(3,1).muliColumnVector(featureScale);
            expectedMax = Nd4j.ones(3,1).muli(samples*timeSteps).muliColumnVector(featureScale);
        }

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
