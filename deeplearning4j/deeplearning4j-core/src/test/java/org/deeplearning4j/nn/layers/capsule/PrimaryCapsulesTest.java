package org.deeplearning4j.nn.layers.capsule;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.PrimaryCapsules;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;

public class PrimaryCapsulesTest extends BaseDL4JTest {

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testOutputType(){
        PrimaryCapsules layer = new PrimaryCapsules.Builder(10, 8)
                .kernelSize(5, 5)
                .stride(4, 4)
                .build();


        InputType in1 = InputType.convolutional(20, 20, 20);
        assertEquals(InputType.recurrent(160, 8), layer.getOutputType(0, in1));

    }

    @Test
    public void testInputType(){
        PrimaryCapsules layer = new PrimaryCapsules.Builder(10, 8)
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
        PrimaryCapsules layer1 = new PrimaryCapsules.Builder(10, 8)
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

        PrimaryCapsules layer2 = new PrimaryCapsules.Builder(10, 8)
                .kernelSize(5, 5)
                .stride(4, 4)
                .build();
        assertFalse(layer2.isUseRelu());

        PrimaryCapsules layer3 = new PrimaryCapsules.Builder(10, 8)
                .kernelSize(5, 5)
                .stride(4, 4)
                .useReLU()
                .build();
        assertTrue(layer3.isUseRelu());
        assertEquals(0, layer3.getLeak());

    }

    //TODO model tests

}
