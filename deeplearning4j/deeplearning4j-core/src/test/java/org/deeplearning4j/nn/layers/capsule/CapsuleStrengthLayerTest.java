package org.deeplearning4j.nn.layers.capsule;

import static org.junit.Assert.assertEquals;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;

public class CapsuleStrengthLayerTest extends BaseDL4JTest {

    @Override
    public DataType getDataType(){
        return DataType.FLOAT;
    }

    @Test
    public void testOutputType(){
        CapsuleStrengthLayer layer = new CapsuleStrengthLayer.Builder().build();

        InputType in1 = InputType.recurrent(5, 8);

        assertEquals(InputType.feedForward(5), layer.getOutputType(0, in1));
    }

    //TODO model tests

}
