package org.deeplearning4j.linalg.ops.elementwise;

import org.deeplearning4j.linalg.ops.TwoArrayElementWiseOp;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test creation of ops
 * @author Adam Gibson
 */
public class CreationTests {

    private static Logger log = LoggerFactory.getLogger(CreationTests.class);


    @Test
    public void testCreation() {
        TwoArrayElementWiseOp add = new TwoArrayOps().from(new DummyNDArray()).to(new DummyNDArray()).op(AddOp.class).build();
        TwoArrayElementWiseOp sub = new TwoArrayOps().from(new DummyNDArray()).to(new DummyNDArray()).op(SubtractOp.class).build();
        TwoArrayElementWiseOp multiply = new TwoArrayOps().from(new DummyNDArray()).to(new DummyNDArray()).op(MultiplyOp.class).build();
        TwoArrayElementWiseOp divide = new TwoArrayOps().from(new DummyNDArray()).to(new DummyNDArray()).op(DivideOp.class).build();

    }



}
