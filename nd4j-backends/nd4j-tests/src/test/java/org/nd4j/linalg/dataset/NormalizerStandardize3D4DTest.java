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


        INDArray labelSet = Nd4j.zeros(samples,3);
        DataSet sampleDataSet = new DataSet(featureMatrix, labelSet);

        myNormalizer.fit(sampleDataSet);
        //assertEquals(myNormalizer.getMean(),Nd4j.create(new float[] {k,k,k}));
        //assertEquals(myNormalizer.getStd(),Nd4j.zeros(1,features));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
