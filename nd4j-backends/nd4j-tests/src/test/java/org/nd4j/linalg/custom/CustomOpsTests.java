package org.nd4j.linalg.custom;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * This class holds various CustomOps tests
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CustomOpsTests {

    @Test
    public void testNonInplaceOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);
        val arrayZ = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .setInputs(arrayX, arrayY)
                .setOutputs(arrayZ)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayZ);
    }

    /**
     * This test works inplace, but without inplace declaration
     */
    @Test
    public void testNonInplaceOp2() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .setInputs(arrayX, arrayY)
                .setOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testNoOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);

        CustomOp op = DynamicCustomOp.builder("noop")
                .setInputs(arrayX, arrayY)
                .setOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);
    }

    @Test
    public void testFloor() throws Exception {
        val arrayX = Nd4j.create(10, 10);

        arrayX.assign(3.0);

        val exp = Nd4j.create(10,10).assign(3.0);

        CustomOp op = DynamicCustomOp.builder("floor")
                .setInputs(arrayX)
                .setOutputs(arrayX)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test
    public void testInplaceOp1() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .setInputs(arrayX, arrayY)
                .callInplace(true)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNoneInplaceOp3() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);

        arrayX.assign(4.0);
        arrayY.assign(2.0);

        val exp = Nd4j.create(10,10).assign(6.0);

        CustomOp op = DynamicCustomOp.builder("add")
                .setInputs(arrayX, arrayY)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
    }


    @Test
    public void testInplaceOp2() throws Exception {
        val arrayX = Nd4j.create(10, 10);
        val arrayY = Nd4j.create(10, 10);
        val arrayZ = Nd4j.create(10, 10);

        arrayX.assign(3.0);
        arrayY.assign(1.0);

        val exp = Nd4j.create(10,10).assign(4.0);
        val expZ = Nd4j.create(10,10);

        CustomOp op = DynamicCustomOp.builder("add")
                .setInputs(arrayX, arrayY)
                .setOutputs(arrayZ)
                .callInplace(true)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, arrayX);
        assertEquals(expZ, arrayZ);
    }


    @Test
    public void testMergeMax1() throws Exception {
        val array0 = Nd4j.create(new double[] {1, 0, 0, 0, 0});
        val array1 = Nd4j.create(new double[] {0, 2, 0, 0, 0});
        val array2 = Nd4j.create(new double[] {0, 0, 3, 0, 0});
        val array3 = Nd4j.create(new double[] {0, 0, 0, 4, 0});
        val array4 = Nd4j.create(new double[] {0, 0, 0, 0, 5});

        val z = Nd4j.create(5);
        val exp = Nd4j.create(new double[]{1, 2, 3, 4, 5});

        CustomOp op = DynamicCustomOp.builder("mergemax")
                .setInputs(array0, array1, array2, array3, array4)
                .setOutputs(z)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        assertEquals(exp, z);
    }
}
