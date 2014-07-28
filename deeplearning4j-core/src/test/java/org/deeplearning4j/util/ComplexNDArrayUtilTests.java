package org.deeplearning4j.util;


import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.Shape;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author Adam Gibson
 */
public class ComplexNDArrayUtilTests {

    private static Logger log = LoggerFactory.getLogger(ComplexNDArrayUtilTests.class);


    @Test
    public void testPadWithZeros() {
        ComplexNDArray ret = new ComplexNDArray(new int[]{2,5});
        ComplexNDArray test = ComplexNDArrayUtil.padWithZeros(ret,new int[]{2,5,5});
        assertEquals(true, Arrays.equals(new int[]{2,5,5},test.shape()));
        assertEquals(ret.sum().real(),test.sum().real(),1e-1);
    }

    @Test
    public void testTruncate() {
        ComplexNDArray ret = new ComplexNDArray(new double[]{1,0,2,0,3,0,4,0},new int[]{2,2});
        //axis 0 or column wise
        ComplexNDArray truncated = ComplexNDArrayUtil.truncate(ret,1,0);
        truncated.toString();
        ComplexNDArray answer = new ComplexNDArray(new double[]{1,0,2,0},new int[]{1,2});
        assertEquals(answer,truncated);
        //axis 1 or row wise
        ComplexNDArray answer0 = new ComplexNDArray(new double[]{1,0,3,0},new int[]{1,2});
        ComplexNDArray truncated0 = ComplexNDArrayUtil.truncate(ret,1,1);
        assertEquals(answer0,truncated0);


        ComplexNDArray arr2 = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,24,24).data,new int[]{4,3,2}));
        ComplexNDArray truncated2 = ComplexNDArrayUtil.truncate(arr2,1,0);
        ComplexNDArray test = new ComplexNDArray(new NDArray(new double[]{1,2,3,4,5,6},new int[]{1,3,2}));
        assertEquals(test, truncated2);

        ComplexNDArray truncated3 = ComplexNDArrayUtil.truncate(arr2,2,0);
        ComplexNDArray testMulti = new ComplexNDArray(new NDArray(new double[]{1,2,3,4,5,6,7,8,9,10,11,12},new int[]{2,3,2}));
        assertEquals(true, Shape.shapeEquals(truncated3.shape(),testMulti.shape()));
        assertEquals(truncated3,testMulti);

        ComplexNDArray thirty = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,30,30).data,new int[]{3,5,2}));
        ComplexNDArray ten = new ComplexNDArray(new NDArray(DoubleMatrix.linspace(1,10,10).data,new int[]{5,2}));

        ComplexNDArray test10 = thirty.slice(0);
        assertEquals(ten,test10);



    }





}
