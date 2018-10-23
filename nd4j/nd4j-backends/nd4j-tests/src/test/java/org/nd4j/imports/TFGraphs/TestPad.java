package org.nd4j.imports.TFGraphs;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TestPad {

    @Test
    public void testPad(){

        INDArray in = Nd4j.create(1, 28, 28, 264);
        INDArray pad = Nd4j.create(new double[][]{{0,0},{0,1},{0,1},{0,0}});
        INDArray out = Nd4j.create(1, 29, 29, 264);

        DynamicCustomOp op = DynamicCustomOp.builder("pad")
                .addInputs(in, pad)
                .addOutputs(out)
                .addIntegerArguments(0) //constant mode, with no constant specified
                .build();

        List<long[]> outShape = Nd4j.getExecutioner().calculateOutputShape(op);
        assertEquals(1, outShape.size());
        assertArrayEquals(new long[]{1, 29, 29, 264}, outShape.get(0));

        Nd4j.getExecutioner().exec(op); //Crash here
    }
}
