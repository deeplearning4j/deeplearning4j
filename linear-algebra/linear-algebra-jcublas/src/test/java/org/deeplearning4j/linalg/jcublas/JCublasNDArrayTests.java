package org.deeplearning4j.linalg.jcublas;

import static org.junit.Assert.*;

import org.deeplearning4j.linalg.factory.NDArrays;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class JCublasNDArrayTests extends org.deeplearning4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(JCublasNDArrayTests.class);

    @Test
    public void testCreation() {
        JCublasNDArray a = (JCublasNDArray) NDArrays.create(new float[]{1,3,2,4,5,6},new int[]{2,3});
        float[] copy = Arrays.copyOf(a.data(),a.length());
        a.alloc();
        a.getData();
        a.free();
        float[] other = a.data();
        assertTrue(Arrays.equals(copy, other));
        a.putScalar(1,1);
        float[] copyTwo = a.data();
        float[] testCopy = Arrays.copyOf(copyTwo,copyTwo.length);
        a.alloc();
        a.getData();
        a.free();
        assertTrue(Arrays.equals(copyTwo, testCopy));



    }

    @Test
    public void testGetSet() {
        JCublasNDArray create = (JCublasNDArray) NDArrays.create(new float[]{1,2,3,4});
        create.alloc();
        create.getData();
        create.free();
        assertTrue(Arrays.equals(new float[]{1,2,3,4},create.data()));


        JCublasNDArray createMatrix = (JCublasNDArray) NDArrays.create(new float[]{1,2,3,4},new int[]{2,2});
        create.alloc();
        createMatrix.data()[0] = 5;
        float[] ret = new float[4];
        createMatrix.getData(ret);
        createMatrix.free();
        assertTrue(Arrays.equals(new float[]{5,2,3,4},ret));


    }


    @Test
    public void testAddColumn() {
        NDArrays.factory().setOrder('f');
        JCublasNDArray a = (JCublasNDArray) NDArrays.create(new float[]{1,3,2,4,5,6},new int[]{2,3});
        JCublasNDArray aDup = (JCublasNDArray) NDArrays.create(new float[]{1.0f,3.0f,3.0f,6.0f,5.0f,6.0f},new int[]{2,3});
        JCublasNDArray column = (JCublasNDArray) a.getColumn(1);
        column.addi(NDArrays.create(new float[]{1, 2}));

        assertEquals(aDup,a);


    }








}