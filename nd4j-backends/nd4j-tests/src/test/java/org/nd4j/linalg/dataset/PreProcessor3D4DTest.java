package org.nd4j.linalg.dataset;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
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

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();

        int timeSteps = 15;
        int samples = 100;
        //multiplier for the features
        INDArray featureScaleA = Nd4j.create(new double [] {1,-2,3}).reshape(3,1);
        INDArray featureScaleB = Nd4j.create(new double [] {2,2,3}).reshape(3,1);

        Construct3dDataSet caseA = new Construct3dDataSet(featureScaleA,timeSteps,samples,1);
        Construct3dDataSet caseB = new Construct3dDataSet(featureScaleB,timeSteps,samples,1);

        myNormalizer.fit(caseA.sampleDataSet);
        assertEquals(caseA.expectedMean, myNormalizer.getMean());
        assertTrue(Transforms.abs(myNormalizer.getStd().div(caseA.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        myMinMaxScaler.fit(caseB.sampleDataSet);
        assertEquals(caseB.expectedMin,myMinMaxScaler.getMin());
        assertEquals(caseB.expectedMax,myMinMaxScaler.getMax());

        //Same Test with an Iterator, values should be close for std, exact for everything else
        DataSetIterator sampleIterA = new TestDataSetIterator(caseA.sampleDataSet,5);
        DataSetIterator sampleIterB = new TestDataSetIterator(caseB.sampleDataSet,5);

        myNormalizer.fit(sampleIterA);
        assertEquals(myNormalizer.getMean(),caseA.expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().div(caseA.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        myMinMaxScaler.fit(sampleIterB);
        assertEquals(myMinMaxScaler.getMin(),caseB.expectedMin);
        assertEquals(myMinMaxScaler.getMax(),caseB.expectedMax);

    }


    @Test
    public void testBruteForce3dMaskLabels() {

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fitLabel(true);
        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fitLabel(true);

        //generating a dataset with consecutive numbers as feature values. Dataset also has masks
        int samples = 100;
        INDArray featureScale = Nd4j.create(new float[] {1,2,10}).reshape(3,1);
        int timeStepsU = 5;
        Construct3dDataSet sampleU = new Construct3dDataSet(featureScale,timeStepsU,samples,1);
        int timeStepsV = 3;
        Construct3dDataSet sampleV = new Construct3dDataSet(featureScale,timeStepsV,samples,sampleU.newOrigin);
        List<DataSet> dataSetList = new ArrayList<DataSet>();
        dataSetList.add(sampleU.sampleDataSet);
        dataSetList.add(sampleV.sampleDataSet);

        DataSet fullDataSetA = DataSet.merge(dataSetList);
        DataSet fullDataSetAA = fullDataSetA.copy();
        //This should be the same datasets as above without a mask
        Construct3dDataSet fullDataSetNoMask = new Construct3dDataSet(featureScale,timeStepsU+timeStepsV,samples,1);

        //preprocessors - label and feature values are the same
        myNormalizer.fit(fullDataSetA);
        assertEquals(myNormalizer.getMean(),fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getStd(),fullDataSetNoMask.expectedStd);
        assertEquals(myNormalizer.getLabelMean(),fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getLabelStd(),fullDataSetNoMask.expectedStd);

        myMinMaxScaler.fit(fullDataSetAA);
        assertEquals(myMinMaxScaler.getMin(),fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getMax(),fullDataSetNoMask.expectedMax);
        assertEquals(myMinMaxScaler.getLabelMin(),fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getLabelMax(),fullDataSetNoMask.expectedMax);


        //Same Test with an Iterator, values should be close for std, exact for everything else
        DataSetIterator sampleIterA = new TestDataSetIterator(fullDataSetA,5);
        DataSetIterator sampleIterB = new TestDataSetIterator(fullDataSetAA,5);

        myNormalizer.fit(sampleIterA);
        assertEquals(myNormalizer.getMean(),fullDataSetNoMask.expectedMean);
        assertEquals(myNormalizer.getLabelMean(),fullDataSetNoMask.expectedMean);
        assertTrue(Transforms.abs(myNormalizer.getStd().div(fullDataSetNoMask.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);
        assertTrue(Transforms.abs(myNormalizer.getLabelStd().div(fullDataSetNoMask.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        myMinMaxScaler.fit(sampleIterB);
        assertEquals(myMinMaxScaler.getMin(),fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getMax(),fullDataSetNoMask.expectedMax);
        assertEquals(myMinMaxScaler.getLabelMin(),fullDataSetNoMask.expectedMin);
        assertEquals(myMinMaxScaler.getLabelMax(),fullDataSetNoMask.expectedMax);
    }

    @Test
    public void testBruteForce4d() {

        //generate samples with this scale
        INDArray samples = Nd4j.create(new double[] {11.1,2.1,10,99,7.156,9}).reshape(1,6);
        //generate channels with this scale
        INDArray channels = Nd4j.create(new double[] {1.1,2,5,4}).reshape(1,4);
        construct4dDataSet imageDataSet = new construct4dDataSet(samples,channels,10,15);

        NormalizerStandardize myNormalizer = new NormalizerStandardize();
        myNormalizer.fit(imageDataSet.sampleDataSet);
        assertEquals(imageDataSet.expectedMean, myNormalizer.getMean());
        assertTrue(Transforms.abs(myNormalizer.getStd().div(imageDataSet.expectedStd).sub(1)).maxNumber().floatValue() < 0.01);

        NormalizerMinMaxScaler myMinMaxScaler = new NormalizerMinMaxScaler();
        myMinMaxScaler.fit(imageDataSet.sampleDataSet);
        assertEquals(imageDataSet.expectedMin, myMinMaxScaler.getMin());
        assertEquals(imageDataSet.expectedMax, myMinMaxScaler.getMax());

        DataSet copyDataSet = imageDataSet.sampleDataSet.copy();
        myNormalizer.transform(copyDataSet);
        //all the channels should have the same value now -> since they are multiples of each other
        //across images (x-k1)/k2 and (3x-k1)/k2
        //difference is 3x/k2 - k1/k2 - x/k2 + k1/k2 = 2x/k2; k2 is the stdDev

        //checks to see if all values are the same
        INDArray transformedVals = copyDataSet.getFeatures();
        INDArray imageUno = transformedVals.slice(0);
        assertEquals(imageUno.slice(0),imageUno.slice(1));
        assertEquals(imageUno.slice(0),imageUno.slice(2));
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

    public class construct4dDataSet{
        /*
            //this is an image - #of images x channels x size x size
            //number of channels is hardcoded as 3
            //the three channels are multiples of each other based on INDArray channels
            //the samples are also multiples of each other based in INDArray samples
        */

        DataSet sampleDataSet;
        INDArray samples,channels;
        int height, width;
        INDArray expectedMean, expectedStd, expectedMin, expectedMax;
        INDArray expectedLabelMean, expectedLabelStd, expectedLabelMin, expectedLabelMax;

        public construct4dDataSet(INDArray samples, INDArray channels, int height, int width) {

            this.samples = samples;
            this.channels = channels;
            this.height = height;
            this.width = width;

            INDArray oneChannel = Nd4j.linspace(1,height*width,height*width).reshape(height,width);
            INDArray imageOne = Nd4j.ones(channels.size(1),height,width);
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(imageOne,oneChannel,imageOne,1,2));
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(imageOne,channels,imageOne,0));

            INDArray allImages = Nd4j.ones(samples.size(1),channels.size(1),height,width);
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(allImages,imageOne,allImages,1,2,3));
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(allImages,samples,allImages,0));

            INDArray labels = Nd4j.linspace(1,samples.size(1),samples.size(1)).reshape(samples.size(1),1);
            sampleDataSet = new DataSet(allImages,labels);

            double templateMean = (height*width+1)/2.0;
            templateMean *= samples.sumNumber().floatValue()/samples.size(1);
            expectedMean = channels.mul(templateMean);

            INDArray calcStd = oneChannel.reshape(1,height*width);
            INDArray tempStd = Nd4j.ones(samples.size(1),height*width);
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(tempStd,calcStd,tempStd,1));
            tempStd = tempStd.mulColumnVector(samples.transpose()).reshape(1,samples.size(1)*height*width);
            float templateStd = tempStd.std(1).getFloat(0,0);
            expectedStd = channels.mul(templateStd);

            expectedLabelMean = labels.mean(0);
            expectedLabelStd = labels.std(0);

            expectedMin = channels.mul(samples.min(1));
            expectedMax = channels.mul(samples.max(1)).mul(height*width);
        }

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
