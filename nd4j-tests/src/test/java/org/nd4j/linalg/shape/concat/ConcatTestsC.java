package org.nd4j.linalg.shape.concat;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class ConcatTestsC extends BaseNd4jTest {
    public ConcatTestsC() {
    }

    public ConcatTestsC(String name) {
        super(name);
    }

    public ConcatTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public ConcatTestsC(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testConcatVertically() {
        INDArray rowVector = Nd4j.ones(5);
        INDArray other = Nd4j.ones(5);
        INDArray concat = Nd4j.vstack(other, rowVector);
        assertEquals(rowVector.rows() * 2, concat.rows());
        assertEquals(rowVector.columns(), concat.columns());

        INDArray arr2 = Nd4j.create(5,5);
        INDArray slice1 = arr2.slice(0);
        INDArray slice2 = arr2.slice(1);
        INDArray arr3 = Nd4j.create(2, 5);
        INDArray vstack = Nd4j.vstack(slice1, slice2);
        assertEquals(arr3,vstack);

        INDArray col1 = arr2.getColumn(0);
        INDArray col2 = arr2.getColumn(1);
        INDArray vstacked = Nd4j.vstack(col1,col2);
        assertEquals(Nd4j.create(4,1),vstacked);



    }

    @Override
    public char ordering() {
        return 'c';
    }
}
