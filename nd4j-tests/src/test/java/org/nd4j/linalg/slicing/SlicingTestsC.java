package org.nd4j.linalg.slicing;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class SlicingTestsC extends BaseNd4jTest  {
    public SlicingTestsC() {
    }

    public SlicingTestsC(String name) {
        super(name);
    }

    public SlicingTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public SlicingTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testSliceShape() {
        INDArray arr = Nd4j.linspace(1,30,30).reshape(3, 5, 2);
        INDArray sliceZero = arr.slice(0);
        INDArray assertion = Nd4j.create(new double[]{1,2,3,4,5,6,7,8,9,10},new int[]{5,2});
        assertArrayEquals(new int[] {5,2},sliceZero.shape());
        assertEquals(assertion,sliceZero);

        INDArray assertionTwo = Nd4j.create(new double[]{11,12,13,14,15,16,17,18,19,20},new int[]{5,2});
        INDArray sliceOne = arr.slice(1);
        assertEquals(assertionTwo,arr.slice(1));
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
