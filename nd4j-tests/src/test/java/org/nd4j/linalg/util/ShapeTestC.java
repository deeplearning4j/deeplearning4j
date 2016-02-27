package org.nd4j.linalg.util;
import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class ShapeTestC extends BaseNd4jTest {

    public ShapeTestC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public ShapeTestC() {
    }

    public ShapeTestC(Nd4jBackend backend) {
        super(backend);
    }

    public ShapeTestC(String name) {
        super(name);
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
    public void testElementWiseCompareOnesInMiddle() {
        INDArray arr = Nd4j.linspace(1,6,6).reshape(2,3);
        INDArray onesInMiddle = Nd4j.linspace(1,6,6).reshape(2,1,3);
        for(int i = 0; i < arr.length(); i++)
            assertEquals(arr.getDouble(i),onesInMiddle.getDouble(i));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
