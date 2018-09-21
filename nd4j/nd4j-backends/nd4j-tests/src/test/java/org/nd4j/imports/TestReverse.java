package org.nd4j.imports;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class TestReverse {

    @Test
    public void testReverse(){

        INDArray in = Nd4j.trueVector(new double[]{1,2,3,4,5,6});
        INDArray out = Nd4j.create(new long[]{6});

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in)
                .addOutputs(out)
                .addIntegerArguments(0)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }

    @Test
    public void testReverse2(){

        INDArray in = Nd4j.trueVector(new double[]{1,2,3,4,5,6});
        INDArray axis = Nd4j.trueScalar(0);
        INDArray out = Nd4j.create(new long[]{6});

        DynamicCustomOp op = DynamicCustomOp.builder("reverse")
                .addInputs(in, axis)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);
    }
}
