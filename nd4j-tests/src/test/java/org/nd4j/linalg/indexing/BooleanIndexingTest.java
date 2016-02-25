package org.nd4j.linalg.indexing;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class BooleanIndexingTest {

    /*
        1D array checks
     */

    @Test
    public void testAnd1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.and(array, Conditions.greaterThan(0.5f)));
    }

    @Test
    public void testAnd2() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.and(array, Conditions.lessThan(6.0f)));
    }

    @Test
    public void testAnd3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.and(array, Conditions.lessThan(5.0f)));
    }

    @Test
    public void testAnd4() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.and(array, Conditions.greaterThan(4.0f)));
    }

    @Test
    public void testAnd5() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertTrue(BooleanIndexing.and(array, Conditions.greaterThanOEqual(1e-5f)));
    }

    @Test
    public void testAnd6() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertFalse(BooleanIndexing.and(array, Conditions.lessThan(1e-5f)));
    }

    @Test
    public void testAnd7() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f, 1e-5f, 1e-5f});

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1e-5f)));
    }

    @Test
    public void testOr1() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.or(array, Conditions.greaterThan(3.0f)));
    }

    @Test
    public void testOr2() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertTrue(BooleanIndexing.or(array, Conditions.lessThan(3.0f)));
    }

    @Test
    public void testOr3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

        assertFalse(BooleanIndexing.or(array, Conditions.greaterThan(6.0f)));
    }

    @Test
    public void testApplyWhere1() throws Exception {
        INDArray array = Nd4j.create(new float[] {-1f, -1f, -1f, -1f, -1f});

        BooleanIndexing.applyWhere(array,Conditions.lessThan(Nd4j.EPS_THRESHOLD),new Value(Nd4j.EPS_THRESHOLD));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(Nd4j.EPS_THRESHOLD)));
    }

    @Test
    public void testApplyWhere2() throws Exception {
        INDArray array = Nd4j.create(new float[] {0f, 0f, 0f, 0f, 0f});

        BooleanIndexing.applyWhere(array,Conditions.lessThan(1.0f),new Value(1.0f));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1.0f)));
    }

    @Test
    public void testApplyWhere3() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-18f, 1e-18f, 1e-18f, 1e-18f, 1e-18f});

        BooleanIndexing.applyWhere(array,Conditions.lessThan(1e-12f),new Value(1e-12f));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.and(array, Conditions.equals(1e-12f)));
    }

    @Test
    public void testApplyWhere4() throws Exception {
        INDArray array = Nd4j.create(new float[] {1e-18f, Float.NaN, 1e-18f, 1e-18f, 1e-18f});

        BooleanIndexing.applyWhere(array,Conditions.lessThan(1e-12f),new Value(1e-12f));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        BooleanIndexing.applyWhere(array,Conditions.isNan(),new Value(1e-16f));

        //System.out.println("Array contains: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-12f)));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-16f)));
    }

    /*
        2D array checks
     */

    @Test
    public void test2dAnd1() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        assertTrue(BooleanIndexing.and(array, Conditions.equals(0f)));
    }

    @Test
    public void test2dAnd2() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);

        assertFalse(BooleanIndexing.and(array, Conditions.equals(0f)));
    }

    @Test
    public void test2dAnd3() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);

        assertFalse(BooleanIndexing.and(array, Conditions.greaterThan(0f)));
    }

    @Test
    public void test2dAnd4() throws Exception {
        INDArray array = Nd4j.zeros(10, 10);

        array.slice(4).putScalar(2, 1e-5f);

        assertTrue(BooleanIndexing.or(array, Conditions.greaterThan(1e-6f)));
    }

    @Test
    public void test2dApplyWhere1() throws Exception {
        INDArray array = Nd4j.ones(4, 4);

        array.slice(3).putScalar(2, 1e-5f);

        //System.out.println("Array before: " + Arrays.toString(array.data().asFloat()));

        BooleanIndexing.applyWhere(array,Conditions.lessThan(1e-4f),new Value(1e-12f));

        //System.out.println("Array after 1: " + Arrays.toString(array.data().asFloat()));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1e-12f)));

        assertTrue(BooleanIndexing.or(array, Conditions.equals(1.0f)));

        assertFalse(BooleanIndexing.and(array, Conditions.equals(1e-12f)));
    }

    /**
     * This test fails, because it highlights current mechanics on SpecifiedIndex stuff.
     * Internally there's
     *
     * @throws Exception
     */
    @Test
    public void testSliceAssign1() throws Exception {
        INDArray array = Nd4j.zeros(4, 4);

        INDArray patch = Nd4j.create(new float[] {1e-5f, 1e-5f, 1e-5f});

        INDArray slice = array.slice(1);
        int[] idx = new int[] {0,1,3};
        INDArrayIndex[] range = new INDArrayIndex[]{new SpecifiedIndex(idx)};

        INDArray subarray = slice.get(range);

        System.out.println("Subarray: " + Arrays.toString(subarray.data().asFloat()) + " isView: " + subarray.isView());

        slice.put(range, patch );

        System.out.println("Array after being patched: " + Arrays.toString(array.data().asFloat()));

        assertFalse(BooleanIndexing.and(array, Conditions.equals(0f)));
    }
}