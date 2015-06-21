package org.nd4j.linalg.cpu.op;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.cpu.ops.NativeOpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

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

}
