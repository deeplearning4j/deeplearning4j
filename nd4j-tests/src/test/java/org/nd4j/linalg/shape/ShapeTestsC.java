package org.nd4j.linalg.shape;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class ShapeTestsC extends BaseNd4jTest {
    @Test
    public void testSixteenZeroOne() {
        INDArray baseArr = Nd4j.linspace(1,16,16).reshape(2, 2, 2, 2);
        assertEquals(4,baseArr.tensorssAlongDimension(0, 1));
        INDArray columnVectorFirst = Nd4j.create(new double[]{1,5,9,13},new int[]{2,2});
        INDArray columnVectorSecond = Nd4j.create(new double[]{2,6,10,14},new int[]{2,2});
        INDArray columnVectorThird = Nd4j.create(new double[]{3,7,11,15},new int[]{2,2});
        INDArray columnVectorFourth = Nd4j.create(new double[]{4,8,12,16},new int[]{2,2});
        INDArray[] assertions = new INDArray[] {
                columnVectorFirst,columnVectorSecond,columnVectorThird,columnVectorFourth
        };
        for(int i = 0; i < baseArr.tensorssAlongDimension(0,1); i++) {
            assertEquals("Wrong at index " + i,assertions[i],baseArr.tensorAlongDimension(i,0,1));
        }

    }

    @Test
    public void testSixteenSecondDim() {
        INDArray baseArr = Nd4j.linspace(1,16,16).reshape(2, 2, 2, 2);
        INDArray[] assertions = new INDArray[] {
                Nd4j.create(new double[]{1,3}),
                Nd4j.create(new double[]{2,4}),
                Nd4j.create(new double[]{5,7}),
                Nd4j.create(new double[]{6,8}),
                Nd4j.create(new double[]{9,11}),
                Nd4j.create(new double[]{10,12}),
                Nd4j.create(new double[]{13,15}),
                Nd4j.create(new double[]{14,16}),


        };

        for(int i = 0; i < baseArr.tensorssAlongDimension(2); i++) {
            INDArray arr = baseArr.tensorAlongDimension(i, 2);
            assertEquals("Failed at index " + i, assertions[i], arr);
        }

    }

    @Test
    public void testThreeTwoTwo() {
        INDArray threeTwoTwo = Nd4j.linspace(1,12,12).reshape(3,2,2);
        INDArray[] assertions = new INDArray[] {
                Nd4j.create(new double[]{1,3}),
                Nd4j.create(new double[]{2,4}),
                Nd4j.create(new double[]{5,7}),
                Nd4j.create(new double[]{6,8}),
                Nd4j.create(new double[]{9,11}),
                Nd4j.create(new double[]{10,12}),

        };

        assertEquals(assertions.length,threeTwoTwo.tensorssAlongDimension(1));
        for(int i = 0; i < assertions.length; i++) {
            assertEquals(assertions[i],threeTwoTwo.tensorAlongDimension(i,1));
        }

    }

    @Test
    public void testThreeTwoTwoTwo() {
        INDArray threeTwoTwo = Nd4j.linspace(1,12,12).reshape(3,2,2);
        INDArray[] assertions = new INDArray[] {
                Nd4j.create(new double[]{1,2}),
                Nd4j.create(new double[]{3,4}),
                Nd4j.create(new double[]{5,6}),
                Nd4j.create(new double[]{7,8}),
                Nd4j.create(new double[]{9,10}),
                Nd4j.create(new double[]{11,12}),

        };

        assertEquals(assertions.length,threeTwoTwo.tensorssAlongDimension(2));
        for(int i = 0; i < assertions.length; i++) {
            assertEquals(assertions[i],threeTwoTwo.tensorAlongDimension(i,2));
        }

    }


    @Test
    public void testSixteenFirstDim() {
        INDArray baseArr = Nd4j.linspace(1,16,16).reshape(2, 2, 2, 2);
        INDArray[] assertions = new INDArray[] {
                Nd4j.create(new double[]{1,5}),
                Nd4j.create(new double[]{2,6}),
                Nd4j.create(new double[]{3,7}),
                Nd4j.create(new double[]{4,8}),
                Nd4j.create(new double[]{9,13}),
                Nd4j.create(new double[]{10,14}),
                Nd4j.create(new double[]{11,15}),
                Nd4j.create(new double[]{12,16}),


        };

        for(int i = 0; i < baseArr.tensorssAlongDimension(1); i++) {
            INDArray arr = baseArr.tensorAlongDimension(i, 1);
            assertEquals("Failed at index " + i, assertions[i], arr);
        }

    }


    @Test
    public void testEight() {
        INDArray baseArr = Nd4j.linspace(1,8,8).reshape(2,2,2);
        assertEquals(2,baseArr.tensorssAlongDimension(0,1));
        INDArray columnVectorFirst = Nd4j.create(new double[]{1,3,5,7}, new int[]{2,2});
        INDArray columnVectorSecond = Nd4j.create(new double[]{2,4,6,8},new int[]{2,2});
        assertEquals(columnVectorFirst,baseArr.tensorAlongDimension(0,0,1));
        assertEquals(columnVectorSecond,baseArr.tensorAlongDimension(1,0,1));

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
