package org.deeplearning4j.nn.layers.capsule;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleLayer;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;

public class CapsuleLayerTest extends BaseDL4JTest {

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testOutputType(){
        CapsuleLayer layer = new CapsuleLayer.Builder(10, 16, 5).build();

        InputType in1 = InputType.recurrent(5, 8);

        assertEquals(InputType.recurrent(10, 16), layer.getOutputType(0, in1));
    }

    @Test
    public void testInputType(){
        CapsuleLayer layer = new CapsuleLayer.Builder(10, 16, 5).build();

        InputType in1 = InputType.recurrent(5, 8);

        layer.setNIn(in1, true);

        assertEquals(5, layer.getInputCapsules());
        assertEquals(8, layer.getInputCapsuleDimensions());
    }

    @Test
    public void testConfig(){
        CapsuleLayer layer1 = new CapsuleLayer.Builder(10, 16, 5).build();

        assertEquals(10, layer1.getCapsules());
        assertEquals(16, layer1.getInputCapsuleDimensions());
        assertEquals(5, layer1.getRoutings());
        assertFalse(layer1.isHasBias());

        CapsuleLayer layer2 = new CapsuleLayer.Builder(10, 16, 5).hasBias(true).build();

        assertTrue(layer1.isHasBias());

    }

    //TODO model tests

}
