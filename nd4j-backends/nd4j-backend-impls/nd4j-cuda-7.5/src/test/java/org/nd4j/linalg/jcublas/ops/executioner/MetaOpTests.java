package org.nd4j.linalg.jcublas.ops.executioner;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.impl.meta.LinearMetaOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class MetaOpTests {

    @Test
    public void testLinearMetaOp1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(10);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Abs opB = new Abs(array);

        LinearMetaOp metaOp = new LinearMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(1).getType());
    }
}
