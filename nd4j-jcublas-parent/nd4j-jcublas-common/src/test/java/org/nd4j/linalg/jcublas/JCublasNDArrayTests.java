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
    public void testAddColumn() {
        Nd4j.factory().setOrder('f');
        JCublasNDArray a = (JCublasNDArray) Nd4j.create(new float[]{1, 3, 2, 4, 5, 6}, new int[]{2, 3});
        JCublasNDArray aDup = (JCublasNDArray) Nd4j.create(new float[]{3.0f,6.0f});
        JCublasNDArray column = (JCublasNDArray) a.getColumn(1);
        column.addi(Nd4j.create(new float[]{1, 2}));

        assertEquals(aDup,column);


    }








}