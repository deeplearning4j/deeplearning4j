package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

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
           This is 3d dataset where dimensions are sample#,feature,timesteps
           Timesteps are set to consecutive nums
           Samples are multiples of each other
           The obtained values are compared to the theoretical mean and std dev
         */
        NormalizerStandardize myNormalizer = new NormalizerStandardize();

        int timeSteps = 12;
        int samples = 4;
        int features = 3;

        float k = 10f;

        INDArray fullFeatures = Nd4j.zeros(samples,features,timeSteps).add(k);
        INDArray labelSet = Nd4j.zeros(samples,features);
        DataSet sampleDataSet = new DataSet(fullFeatures, labelSet);

        myNormalizer.fit(sampleDataSet);
        assertEquals(myNormalizer.getMean(),Nd4j.create(new float[] {k,k,k}));
        assertEquals(myNormalizer.getStd(),Nd4j.zeros(1,features));
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
