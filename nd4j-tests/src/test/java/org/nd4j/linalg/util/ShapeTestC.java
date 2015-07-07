package org.nd4j.linalg.util;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class ShapeTestC extends BaseNd4jTest {

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
        assertEquals(getTensor,getTensorZero);


    }
    
    @Override
    public char ordering() {
        return 'c';
    }
}
