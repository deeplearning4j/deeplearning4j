package org.nd4j.linalg.util;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

/**
 * @author Adam Gibson
 */
public class ShapeTest extends BaseNd4jTest {
    public ShapeTest() {
    }

    public ShapeTest(Nd4jBackend backend) {
        super(backend);
    }

    public ShapeTest(String name) {
        super(name);
    }

    public ShapeTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Test
    public void testToOffsetZero() {
        INDArray matrix  =  Nd4j.rand(3,5);
        INDArray rowOne = matrix.getRow(1);
        INDArray row1Copy = Shape.toOffsetZero(rowOne);
        assertEquals(rowOne,row1Copy);
        INDArray rows =  matrix.getRows(1, 2);
        INDArray rowsOffsetZero = Shape.toOffsetZero(rows);
        assertEquals(rows,rowsOffsetZero);

        INDArray tensor = Nd4j.rand(new int[]{3,3,3});
        INDArray getTensor = tensor.slice(1).slice(1);
        INDArray getTensorZero = Shape.toOffsetZero(getTensor);
        assertEquals(getTensor, getTensorZero);



    }


    @Test
    public void testDupLeadingTrailingZeros() {
        testDupHelper(1,10);
        testDupHelper(10,1);
        testDupHelper(1, 10, 1);
        testDupHelper(1, 10, 1, 1);
        testDupHelper(1,10,2);
        testDupHelper(2, 10, 1, 1);
        testDupHelper(1, 1, 1, 10);
        testDupHelper(10, 1, 1, 1);
        testDupHelper(1,1);

    }

    private  void testDupHelper(int... shape) {
        INDArray arr = Nd4j.ones(shape);
        INDArray arr2 = arr.dup();
        assertArrayEquals(arr.shape(), arr2.shape());
        assertTrue(arr.equals(arr2));
    }

    @Test
    public void testLeadingOnes() {
        INDArray arr = Nd4j.create(1, 5, 5);
        assertEquals(1,arr.getLeadingOnes());
        INDArray arr2 = Nd4j.create(2, 2);
        assertEquals(0, arr2.getLeadingOnes());
        INDArray arr4 = Nd4j.create(1,1,5,5);
        assertEquals(2, arr4.getLeadingOnes());
    }

    @Test
    public void testTrailingOnes() {
        INDArray arr2 = Nd4j.create(5,5,1);
        assertEquals(1,arr2.getTrailingOnes());
        INDArray arr4 = Nd4j.create(5, 5, 1, 1);
        assertEquals(2,arr4.getTrailingOnes());
    }

    @Test
    public void testElementWiseCompareOnesInMiddle() {
        INDArray arr = Nd4j.linspace(1,6,6).reshape(2, 3);
        INDArray onesInMiddle = Nd4j.linspace(1,6,6).reshape(2, 1, 3);
        for(int i = 0; i < arr.length(); i++) {
            double val = arr.getDouble(i);
            double middleVal = onesInMiddle.getDouble(i);
            assertEquals(val,middleVal);
        }
    }


    @Test
    public void testSumLeadingTrailingZeros(){
        testSumHelper(1,5,5);
        testSumHelper(5,5,1);
        testSumHelper(1,5,1);

        testSumHelper(1,5,5,5);
        testSumHelper(5,5,5,1);
        testSumHelper(1,5,5,1);

        testSumHelper(1,5,5,5,5);
        testSumHelper(5,5,5,5,1);
        testSumHelper(1,5,5,5,1);

        testSumHelper(1,5,5,5,5,5);
        testSumHelper(5, 5, 5, 5, 5, 1);
        testSumHelper(1, 5, 5, 5, 5, 1);
    }

    private  void testSumHelper( int... shape ) {
        INDArray array = Nd4j.ones(shape);
        for( int i = 0; i < shape.length; i++) {
            for(int j = 0; j < array.vectorsAlongDimension(i); j++) {
                INDArray vec = array.vectorAlongDimension(j,i);
            }
            array.sum(i);
        }
    }




    @Override
    public char ordering() {
        return 'f';
    }
}
