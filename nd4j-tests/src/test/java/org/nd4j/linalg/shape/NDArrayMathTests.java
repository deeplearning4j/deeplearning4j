package org.nd4j.linalg.shape;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NDArrayMath;


/**
 * @author Adam Gibson
 */
public class NDArrayMathTests extends BaseNd4jTest {
    @Test
    public void testVectorPerSlice() {
        INDArray arr = Nd4j.create(2,2,2,2);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arr));

        INDArray matrix = Nd4j.create(2,2);
        assertEquals(1,NDArrayMath.vectorsPerSlice(matrix));

        INDArray arrSliceZero = arr.slice(0);
        assertEquals(4,NDArrayMath.vectorsPerSlice(arrSliceZero));

    }

    @Test
    public void testMatricesPerSlice() {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(2,NDArrayMath.matricesPerSlice(arr));
    }

    @Test
    public void testLengthPerSlice() {
        INDArray arr = Nd4j.create(2,2,2,2);
        int lengthPerSlice = NDArrayMath.lengthPerSlice(arr);
        assertEquals(8,lengthPerSlice);
    }

    @Test
    public void testSliceForVector() {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(0,NDArrayMath.sliceForVector(1,arr,0));
        assertEquals(1,NDArrayMath.sliceForVector(4,arr,0));
    }

    @Test
    public void toffsetForSlice() {
        INDArray arr = Nd4j.create(3,2,2);
        int slice = 1;
        assertEquals(4,NDArrayMath.offsetForSlice(arr,slice));
    }

    @Test
    public void testSliceForVectorOffset() {
        INDArray arr = Nd4j.create(3,2,2);
        assertEquals(0,NDArrayMath.sliceForVector(1,arr,0));
        assertEquals(1,NDArrayMath.sliceForVector(2,arr,0));
    }

    @Test
    public void testMapOntoVector() {
        INDArray arr = Nd4j.create(3,2,2);
        assertEquals(NDArrayMath.mapIndexOntoVector(2,arr),4);
    }

    @Test
    public void testNumVectors() {
        INDArray arr = Nd4j.create(3,2,2);
        assertEquals(4,NDArrayMath.vectorsPerSlice(arr));
        INDArray matrix = Nd4j.create(2,2);
        assertEquals(1,NDArrayMath.vectorsPerSlice(matrix));

    }


    @Test
    public void testOddDimensions() {
        INDArray arr = Nd4j.create(3,2,2);
        int numMatrices = NDArrayMath.matricesPerSlice(arr);
        assertEquals(1,numMatrices);
    }

    @Test
    public void testTotalVectors() {
        INDArray arr2 = Nd4j.create(2, 2, 2, 2);
        assertEquals(8,NDArrayMath.numVectors(arr2));
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
