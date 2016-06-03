package org.nd4j.linalg.dataset;


import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.BasicDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;


/**
 * Created by susaneraly on 5/25/16.
 */
@RunWith(Parameterized.class)
public class NormalizerMinMaxScalerTest  extends BaseNd4jTest {

    public NormalizerMinMaxScalerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBruteForce() {
        //X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        //X_scaled = X_std * (max - min) + min
        // Dataset features are scaled consecutive natural numbers
        int nSamples = 500;
        int x = 4,y = 2, z = 3;

        INDArray featureX = Nd4j.linspace(1,nSamples,nSamples).reshape(nSamples,1);
        INDArray featureY = featureX.mul(y);
        INDArray featureZ = featureX.mul(z);
        featureX.muli(x);
        INDArray featureSet = Nd4j.concat(1,featureX,featureY,featureZ);
        INDArray labelSet = Nd4j.zeros(nSamples, 1);
        DataSet sampleDataSet = new DataSet(featureSet, labelSet);

        //expected min and max
        INDArray theoreticalMin = Nd4j.create(new double[] {x,y,z});
        INDArray theoreticalMax = Nd4j.create(new double[] {nSamples*x,nSamples*y,nSamples*z});
        INDArray theoreticalRange = theoreticalMax.sub(theoreticalMin);

        NormalizerMinMaxScaler myNormalizer = new NormalizerMinMaxScaler();
        myNormalizer.fit(sampleDataSet);

        INDArray minDataSet = myNormalizer.getMin();
        INDArray maxDataSet = myNormalizer.getMax();
        assertEquals(minDataSet.sub(theoreticalMin).max(1).getDouble(0,0), 0.0, 0.000000001);
        assertEquals(maxDataSet.sub(theoreticalMax).max(1).getDouble(0,0), 0.0, 0.000000001);

        // SAME TEST WITH THE ITERATOR
        int bSize = 1;
        DataSetIterator sampleIter = new BasicDataSetIterator(sampleDataSet,bSize);
        myNormalizer.fit(sampleIter);
        minDataSet = myNormalizer.getMin();
        maxDataSet = myNormalizer.getMax();
        assertEquals(minDataSet.sub(theoreticalMin).max(1).getDouble(0,0), 0.0, 0.000000001);
        assertEquals(maxDataSet.sub(theoreticalMax).max(1).getDouble(0,0), 0.0, 0.000000001);

        sampleIter.setPreProcessor(myNormalizer);
        INDArray actual,expected,delta;
        int i = 1;
        while (sampleIter.hasNext()) {
            expected = theoreticalMin.mul(i-1).div(theoreticalRange);
            actual = sampleIter.next().getFeatures();
            delta = Transforms.abs(actual.sub(expected));
            assertTrue(delta.max(1).getDouble(0,0) < 0.0001);
            i++;
        }

    }

    @Override
    public char ordering() {
        return 'c';
    }
}

