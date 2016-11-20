package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.*;


/**
 * Created by susaneraly on 6/23/16.
 */
@RunWith(Parameterized.class)
public class ImagePreProcessortTest extends BaseNd4jTest {
    public ImagePreProcessortTest (Nd4jBackend backend) {
        super(backend);
    }
    @Test
    public void simpleImageTest() {
        INDArray rChannels = Nd4j.zeros(10,10).addi(128);
        INDArray gChannels = Nd4j.zeros(10,10).addi(64);
        INDArray bChannels = Nd4j.zeros(10,10).addi(255);
        INDArray image = Nd4j.vstack(rChannels,gChannels,bChannels).reshape(3,10,10);
        INDArray orig = image.dup();

        //System.out.println(Arrays.toString(image.shape()));
        DataSet ds = new DataSet(image.reshape(1,3,10,10),Nd4j.ones(1,1));
        ImagePreProcessingScaler myScaler = new ImagePreProcessingScaler();
        //So this should scale to 0.5,0.25 and 1;
        INDArray expected = image.mul(0);
        expected.slice(0,0).addi(0.5);
        expected.slice(1,0).addi(0.25);
        expected.slice(2,0).addi(1.0);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(expected)).maxNumber().doubleValue() <= 0.01 );

        //Now giving it 16 bits instead of the default
        //System.out.println(Arrays.toString(image.shape()));
        ds = new DataSet(image.reshape(1,3,10,10),Nd4j.ones(1,1));
        myScaler = new ImagePreProcessingScaler(0,1,16);
        //So this should scale to 0.5,0.25 and 1;
        expected = image.mul(0);
        expected.slice(0,0).addi(0.5/256);
        expected.slice(1,0).addi(0.25/256);
        expected.slice(2,0).addi(1.0/256);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(expected)).maxNumber().doubleValue() <= 0.01 );

        //So this should not change the value
        INDArray before = ds.getFeatures().dup();
        myScaler = new ImagePreProcessingScaler(0,1,1);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(before)).maxNumber().doubleValue() <= 0.0001 );

        //Scaling back up should give the same results
        myScaler = new ImagePreProcessingScaler(0,(256*256*256-1),1);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(image)).maxNumber().doubleValue() <= 1 );

        //Revert:
        before = orig.dup();
        myScaler = new ImagePreProcessingScaler(0,1,1);
        myScaler.transform(before);
        myScaler.revertFeatures(before);
        assertEquals(orig, before);

    }
    @Override
    public char ordering() {
        return 'c';
    }
}
