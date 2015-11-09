package org.nd4j.linalg.cpu.op;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.cpu.ops.NativeOpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class NativeOpExecutionerTest {

    @Test
    public void testExp() {
        OpExecutioner op = new NativeOpExecutioner();
        INDArray op2 = op.execAndReturn(new Exp(Nd4j.linspace(1,4,4),Nd4j.linspace(1,4,4)));
        System.out.println(op2);
    }

    @Test
    public void testTadOutput() {
        INDArray arr = Nd4j.create(2,10,10,10,10);
        System.out.println("Shape for dimension 0 " + Arrays.toString(arr.tensorAlongDimension(0, 0).shape()) + Arrays.toString(arr.tensorAlongDimension(0, 0).stride()) + " and tads " + arr.tensorssAlongDimension(0));
        System.out.println("Shape for dimension 1 " + Arrays.toString(arr.tensorAlongDimension(1, 1).shape()) + Arrays.toString(arr.tensorAlongDimension(1, 1).stride()) + " and tads " + arr.tensorssAlongDimension(1));
        System.out.println("Length " + arr.tensorAlongDimension(0,1).length());

    }
}
