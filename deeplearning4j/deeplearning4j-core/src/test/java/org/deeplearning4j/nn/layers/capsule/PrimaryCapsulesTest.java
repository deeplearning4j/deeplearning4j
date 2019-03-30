package org.deeplearning4j.nn.layers.capsule;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleLayer;
import org.deeplearning4j.nn.conf.layers.PrimaryCapsules;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PrimaryCapsulesTest extends BaseDL4JTest {

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testOutputType(){
        PrimaryCapsules layer = new PrimaryCapsules.Builder(8, 10)
                .kernelSize(5, 5)
                .stride(4, 4)
                .build();


        InputType in1 = InputType.convolutional(20, 20, 20);
        assertEquals(InputType.recurrent(160, 8), layer.getOutputType(0, in1));

    }

    @Test
    public void testInputType(){
        PrimaryCapsules layer = new PrimaryCapsules.Builder(8, 10)
                .kernelSize(5, 5)
                .stride(4, 4)
                .build();
        InputType in1 = InputType.convolutional(20, 20, 20);


        layer.setNIn(in1, true);

        assertEquals(160, layer.getCapsules());
        assertEquals(8, layer.getCapsuleDimensions());
    }

    @Test
    public void testConfig(){
        PrimaryCapsules layer1 = new PrimaryCapsules.Builder(8, 10)
                .kernelSize(5, 5)
                .stride(4, 4)
                .useLeakyReLU(0.5)
                .build();

        assertEquals(10, layer1.getCapsuleDimensions());
        assertEquals(8, layer1.getChannels());
        assertArrayEquals(new int[]{5, 5}, layer1.getKernelSize());
        assertArrayEquals(new int[]{4, 4}, layer1.getStride());
        assertArrayEquals(new int[]{0, 0}, layer1.getPadding());
        assertArrayEquals(new int[]{1, 1}, layer1.getDilation());
        assertTrue(layer1.isUseRelu());
        assertEquals(0.2, layer1.getLeak());

        PrimaryCapsules layer2 = new PrimaryCapsules.Builder(8, 10)
                .kernelSize(5, 5)
                .stride(4, 4)
                .build();
        assertFalse(layer2.isUseRelu());

        PrimaryCapsules layer3 = new PrimaryCapsules.Builder(8, 10)
                .kernelSize(5, 5)
                .stride(4, 4)
                .useReLU()
                .build();
        assertTrue(layer3.isUseRelu());
        assertEquals(0, layer3.getLeak());

    }

    @Test
    public void testLayer(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .list()
                .layer(new PrimaryCapsules.Builder(8, 10)
                        .kernelSize(5, 5)
                        .stride(4, 4)
                        .useLeakyReLU(0.5)
                        .build())
                .setInputType(InputType.convolutional(20, 20, 20))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(64, 20, 20, 20);

        long[] shape = model.output(emptyFeatures).shape();

        assertArrayEquals(new long[]{64, 160, 8}, shape);
    }

}
