package org.nd4j.linalg.jcublas;

import static org.junit.Assert.*;

import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;


/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class JCublasNDArrayTests extends org.nd4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(JCublasNDArrayTests.class);

    @Test
    public void testCreation() {
        JCublasNDArray a = (JCublasNDArray) Nd4j.create(new float[]{1, 3, 2, 4, 5, 6}, new int[]{2, 3});
        float[] copy = Arrays.copyOf(a.data(),a.length());
        a.alloc();
        a.getData();
        a.free();
        float[] other = a.data();
        assertTrue(Arrays.equals(copy, other));
        a.putScalar(1,1);
        float[] copyTwo = a.data();
        a.alloc();
        float[] data = new float[copyTwo.length];
        a.getData(data);
        a.free();
        assertTrue(Arrays.equals(a.data(), data));



    }

    @Test
    public void testGetSet() {
        JCublasNDArray create = (JCublasNDArray) Nd4j.create(new float[]{1, 2, 3, 4});
        create.alloc();
        create.getData();

        //create.free();
        assertTrue(Arrays.equals(new float[]{1,2,3,4},create.data()));


        JCublasNDArray createMatrix = (JCublasNDArray) Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        createMatrix.data()[0] = 5;
        createMatrix.alloc();
        float[] ret = new float[4];
        createMatrix.getData(ret);
       // createMatrix.free();
        assertTrue(Arrays.equals(new float[]{5,2,3,4},ret));

        JCublasNDArray r = (JCublasNDArray) createMatrix.getRow(1);
        r.alloc();
        //data is only 2 in length: should be 3,4
        float[] d  = new float[2];
        r.getData(d);
        r.free();
        assertTrue(Arrays.equals(new float[]{3,4},d));


    }


    @Test
    public void testAddColumn() {
        Nd4j.factory().setOrder('f');
        JCublasNDArray a = (JCublasNDArray) Nd4j.create(new float[]{1, 3, 2, 4, 5, 6}, new int[]{2, 3});
        JCublasNDArray aDup = (JCublasNDArray) Nd4j.create(new float[]{3.0f,6.0f});
        JCublasNDArray column = (JCublasNDArray) a.getColumn(1);
        column.addi(Nd4j.create(new float[]{1, 2}));

        assertEquals(aDup,column);


    }








}