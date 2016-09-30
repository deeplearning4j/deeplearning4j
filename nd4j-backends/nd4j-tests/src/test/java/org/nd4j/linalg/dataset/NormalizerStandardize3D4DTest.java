package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
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
public class NormalizerStandardize3D4DTest  extends BaseNd4jTest {

    public NormalizerStandardize3D4DTest(Nd4jBackend backend) {
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

        int timeSteps = 12;
        int samples = 10;
        //multiplier for the features
        INDArray featureABC = Nd4j.create(3,1);
        featureABC.put(0,0,2);
        featureABC.put(1,0,-2);
        featureABC.put(2,0,3);

        INDArray template = Nd4j.linspace(1,timeSteps,timeSteps);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template.muliColumnVector(featureABC);
        template = template.reshape(1,3,timeSteps);
        INDArray featureMatrix = template.dup();
        // /\ /\ i = 0
        //the rest \/\/
        for (int i=1;i<samples;i++) {
            template = Nd4j.linspace(i*timeSteps+1,(i+1)*timeSteps,timeSteps);
            template = Nd4j.concat(0,Nd4j.linspace(i*timeSteps+1,(i+1)*timeSteps,timeSteps),template);
            template = Nd4j.concat(0,Nd4j.linspace(i*timeSteps+1,(i+1)*timeSteps,timeSteps),template);
            template.muliColumnVector(featureABC);
            template = template.reshape(1,3,timeSteps);
            featureMatrix = Nd4j.concat(0,featureMatrix,template);
        }


        INDArray labelSet = Nd4j.zeros(samples,3,timeSteps);
        DataSet sampleDataSet = new DataSet(featureMatrix, labelSet);

        myNormalizer.fit(sampleDataSet);
        // The theoretical mean should be the mean of 1,..samples*timesteps
        float theoreticalMean = (samples*timeSteps + 1)/2.0f;
        INDArray expectedMean = Nd4j.create(new float[] {theoreticalMean,theoreticalMean,theoreticalMean}).reshape(3,1);
        expectedMean.muliColumnVector(featureABC);
        float stdNaturalNums = (float) Math.sqrt((samples*samples*timeSteps*timeSteps - 1)/12);
        INDArray expectedStd = Nd4j.create(new float[] {stdNaturalNums,stdNaturalNums,stdNaturalNums}).reshape(3,1);
        expectedStd.muliColumnVector(featureABC);
        //assertEquals(myNormalizer.getMean(),expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getMean().sub(expectedMean)).div(expectedMean).maxNumber().floatValue() < 0.03f);
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd)).div(expectedStd).maxNumber().floatValue() < 0.03f);
    }


    @Test
    public void testBruteForce3dMask() {
        /*
           This is 3d dataset where dimensions are sample#,features,timesteps
           Timesteps are set to consecutive nums continuing across samples
           Each feature is a multiple of the other. Yes they are collinear but this is a test :)
           The obtained values are compared to the theoretical mean and std dev
         */
        NormalizerStandardize myNormalizer = new NormalizerStandardize();

        int timeSteps = 5;
        int totalTimeSteps = timeSteps;
        int samples = 100;
        //multiplier for the features
        INDArray featureABC = Nd4j.create(3,1);
        featureABC.put(0,0,1);
        featureABC.put(1,0,2);
        featureABC.put(2,0,10);

        INDArray template = Nd4j.linspace(1,timeSteps,timeSteps);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template = Nd4j.concat(0,Nd4j.linspace(1,timeSteps,timeSteps),template);
        template.muliColumnVector(featureABC);
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
            template.muliColumnVector(featureABC);
            template = template.reshape(1,3,timeSteps);
            newStart = newEnd + 1;
            featureMatrix = Nd4j.concat(0,featureMatrix,template);
        }
        INDArray labelSet = Nd4j.zeros(samples,3,timeSteps);
        DataSet sampleDataSetU = new DataSet(featureMatrix, labelSet);


        //Continuing on with a new dataset to merge with the one above
        timeSteps = 3;
        totalTimeSteps += timeSteps;
        newEnd = newStart + timeSteps - 1;
        template = Nd4j.linspace(newStart,newEnd,timeSteps);
        template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
        template = Nd4j.concat(0,Nd4j.linspace(newStart,newEnd,timeSteps),template);
        template.muliColumnVector(featureABC);
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
            template.muliColumnVector(featureABC);
            template = template.reshape(1,3,timeSteps);
            newStart = newEnd + 1;
            featureMatrix = Nd4j.concat(0,featureMatrix,template);
        }
        labelSet = Nd4j.zeros(samples,3,timeSteps);
        DataSet sampleDataSetV = new DataSet(featureMatrix, labelSet);


        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(sampleDataSetU);
        dataSetList.add(sampleDataSetV);
        DataSet fullDataSetUV = DataSet.merge(dataSetList);
        myNormalizer.fit(fullDataSetUV);
        // The theoretical mean should be the mean of 1,..samples*timesteps
        int maxN = samples*totalTimeSteps;
        float theoreticalMean = (maxN + 1)/2.0f;
        INDArray expectedMean = Nd4j.create(new float[] {theoreticalMean,theoreticalMean,theoreticalMean}).reshape(3,1);
        expectedMean.muliColumnVector(featureABC);

        float stdNaturalNums = (float) Math.sqrt((maxN*maxN- 1)/12);
        INDArray expectedStd = Nd4j.create(new float[] {stdNaturalNums,stdNaturalNums,stdNaturalNums}).reshape(3,1);
        expectedStd.muliColumnVector(featureABC);
        //std calculates the sample std so divides by (n-1) not n
        expectedStd.muli(Math.sqrt(maxN)).divi(Math.sqrt(maxN-1));

        assertEquals(myNormalizer.getMean(),expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        System.out.println("Actual std");
        System.out.println(myNormalizer.getStd());
        System.out.println("Expected std");
        System.out.println(expectedStd);

        //Same Test with an Iterator
        DataSetIterator sampleIter = new TestDataSetIterator(fullDataSetUV,5);
        myNormalizer.fit(sampleIter);
        System.out.println("Testing with an iterator...");
        System.out.println("Testing mean with an iterator...");
        assertEquals(myNormalizer.getMean(),expectedMean);
        System.out.println("Testing std with an iterator...");
        assertTrue(Transforms.abs(myNormalizer.getStd().sub(expectedStd).div(expectedStd)).maxNumber().floatValue() < 0.03);
        System.out.println("Actual std...");
        System.out.println(myNormalizer.getStd());
        System.out.println("Expected std...");
        System.out.println(expectedStd);

    }

    @Test
    public void transform3d() {
        // A single sample 2 features, 5 timesteps
        INDArray singleSample = Nd4j.linspace(1,10,10).reshape(1,2,5);
        INDArray allFeatures = Nd4j.concat(0,singleSample,singleSample,singleSample,singleSample,singleSample);
        allFeatures = Nd4j.concat(0,allFeatures,allFeatures);
        allFeatures = Nd4j.concat(0,allFeatures,allFeatures);
        allFeatures = Nd4j.concat(0,allFeatures,allFeatures);
        INDArray allLabels = Nd4j.create(40,2,5);
        DataSet dataSet = new DataSet(allFeatures,allLabels);

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fit(dataSet);

        INDArray theMean = Nd4j.create(new float[] {3.0f,8.0f},new int[] {1,2});
        INDArray theStd = Nd4j.create(new float[] {(float) Math.sqrt(2), (float) (Math.sqrt(2))}, new int[] {1,2});
        INDArray expected = singleSample.reshape(2,5).subiColumnVector(theMean.transpose());
        expected.diviColumnVector(theStd.transpose());
        myNormalizer.transform(dataSet);
        assertEquals(expected.toString(),dataSet.getFeatures().slice(0,0).toString());
    }

    @Test
    public void testBruteForce4d() {
        //this is an image - #of images x channels x size x size
        // test with 2 samples, 3 channels x 10 x 10
        // the two samples are multiples of each other 1:3
        INDArray oneChannel = Nd4j.linspace(1,100,100).reshape(1,10,10);
        INDArray imageOne = Nd4j.concat(0,oneChannel,oneChannel.mul(2),oneChannel.mul(3)).reshape(1,3,10,10);
        INDArray imageTwo = imageOne.mul(3);

        INDArray allImages = Nd4j.concat(0,imageOne,imageTwo);
        INDArray labels = Nd4j.create(2,1);

        DataSet dataSet = new DataSet(allImages,labels);
        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fit(dataSet);

        //works out to be 1->100 and 3*(1->100)
        //mean is 4*(1->100)/200  = (100*101/2)/50 = 101
        // std is 82.1599

        INDArray expectedMean = Nd4j.linspace(1,3,3).reshape(1,3).mul(101);
        INDArray expectedStd = Nd4j.linspace(1,3,3).reshape(1,3).mul(82.1599);
        System.out.println("Actual std");
        System.out.println(myNormalizer.getStd());
        System.out.println("Expected std");
        System.out.println(expectedStd);

        assertTrue(Transforms.abs(expectedMean.sub(myNormalizer.getMean()).div(expectedMean)).maxNumber().floatValue() < 0.03f);
        assertTrue(Transforms.abs(expectedStd.sub(myNormalizer.getStd()).div(expectedStd)).maxNumber().floatValue() < 0.03f);

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
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
