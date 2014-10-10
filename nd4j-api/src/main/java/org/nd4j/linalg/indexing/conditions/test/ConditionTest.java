package org.nd4j.linalg.indexing.conditions.test;

import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Created by agibsonccc on 10/10/14.
 */
public abstract class ConditionTest {

    @Test
    public void testNeq() {
        INDArray n = Nd4j.create(new float[]{1,2,3,4});
        INDArray n2 = n.neq(Nd4j.create(new int[]{4}));
        assertEquals(4,n2.sum(Integer.MAX_VALUE).get(0),1e-1);

    }
    @Test
    public void testEq() {
        INDArray n = Nd4j.create(new float[]{1,2,3,4});
        INDArray n2 = n.eq(Nd4j.create(new int[]{4}));
        assertEquals(0,n2.sum(Integer.MAX_VALUE).get(0),1e-1);

    }

}
