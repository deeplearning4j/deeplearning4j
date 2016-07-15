package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

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

        double tolerancePerc = 0.01; // 0.01% of correct value
        int timeSteps = 12;
        int samples = 4;
        int features = 3;
        int x = 2,y = 4, z = 3;
        int a = 2,b = 3,c= 4;

        /*
        INDArray featureX = Nd4j.linspace(1,timeSteps,timeSteps).reshape(1,timeSteps).mul(x);
        INDArray featureY = featureX.mul(y);
        INDArray featureZ = featureX.mul(z);
        INDArray featureSet = Nd4j.concat(0,featureX,featureY,featureZ); //this is one sample
        */

        //INDArray fullFeatures = Nd4j.concat(0,featureSet,featureSet.mul(a),featureSet.mul(b),featureSet.mul(c));
        INDArray fullFeatures = Nd4j.zeros(samples,features,timeSteps).add(10);
        INDArray labelSet = Nd4j.zeros(samples,features);
        DataSet sampleDataSet = new DataSet(fullFeatures, labelSet);

        //double meanNaturalNums = (nSamples + 1)/2.0;
        //INDArray theoreticalMean = Nd4j.create(new double[] {meanNaturalNums*x,meanNaturalNums*y,meanNaturalNums*z});
        //double stdNaturalNums = Math.sqrt((nSamples*nSamples - 1)/12.0);
        //INDArray theoreticalStd = Nd4j.create(new double[] {stdNaturalNums*x,stdNaturalNums*y,stdNaturalNums*z});
    }
    @Override
    public char ordering() {
        return 'c';
    }
}
