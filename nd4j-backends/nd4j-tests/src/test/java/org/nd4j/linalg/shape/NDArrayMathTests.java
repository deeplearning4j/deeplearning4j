package org.nd4j.linalg.shape;

import lombok.val;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.NDArrayMath;

import static org.junit.Assert.assertEquals;


/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class NDArrayMathTests extends BaseNd4jTest {

    public NDArrayMathTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testVectorPerSlice() {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arr));

        INDArray matrix = Nd4j.create(2, 2);
        assertEquals(2, NDArrayMath.vectorsPerSlice(matrix));

        INDArray arrSliceZero = arr.slice(0);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arrSliceZero));

    }

    @Test
    public void testMatricesPerSlice() {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        assertEquals(2, NDArrayMath.matricesPerSlice(arr));
    }

    @Test
    public void testLengthPerSlice() {
        INDArray arr = Nd4j.create(2, 2, 2, 2);
        val lengthPerSlice = NDArrayMath.lengthPerSlice(arr);
        assertEquals(8, lengthPerSlice);
    }

    @Test
    public void toffsetForSlice() {
        INDArray arr = Nd4j.create(3, 2, 2);
        int slice = 1;
        assertEquals(4, NDArrayMath.offsetForSlice(arr, slice));
    }


    @Test
    public void testMapOntoVector() {
        INDArray arr = Nd4j.create(3, 2, 2);
        assertEquals(NDArrayMath.mapIndexOntoVector(2, arr), 4);
    }

    @Test
    public void testNumVectors() {
        INDArray arr = Nd4j.create(3, 2, 2);
        assertEquals(4, NDArrayMath.vectorsPerSlice(arr));
        INDArray matrix = Nd4j.create(2, 2);
        assertEquals(2, NDArrayMath.vectorsPerSlice(matrix));

    }

    @Test
    public void testOffsetForSlice() {
        INDArray arr = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        int[] dimensions = {0, 1};
        INDArray permuted = arr.permute(2, 3, 0, 1);
        int[] test = {0, 0, 1, 1};
        for (int i = 0; i < permuted.tensorssAlongDimension(dimensions); i++) {
            assertEquals(test[i], NDArrayMath.sliceOffsetForTensor(i, permuted, new int[] {2, 2}));
        }

        val arrTensorsPerSlice = NDArrayMath.tensorsPerSlice(arr, new int[] {2, 2});
        assertEquals(2, arrTensorsPerSlice);

        INDArray arr2 = Nd4j.linspace(1, 12, 12).reshape(3, 2, 2);
        int[] assertions = {0, 1, 2};
        for (int i = 0; i < assertions.length; i++) {
            assertEquals(assertions[i], NDArrayMath.sliceOffsetForTensor(i, arr2, new int[] {2, 2}));
        }



        val tensorsPerSlice = NDArrayMath.tensorsPerSlice(arr2, new int[] {2, 2});
        assertEquals(1, tensorsPerSlice);


        INDArray otherTest = Nd4j.linspace(1, 144, 144).reshape(6, 3, 2, 2, 2);
        System.out.println(otherTest);
        INDArray baseArr = Nd4j.linspace(1, 8, 8).reshape(2, 2, 2);
        for (int i = 0; i < baseArr.tensorssAlongDimension(0, 1); i++) {
            System.out.println(NDArrayMath.sliceOffsetForTensor(i, baseArr, new int[] {2, 2}));
        }


    }

    @Test
    public void testOddDimensions() {
        INDArray arr = Nd4j.create(3, 2, 2);
        val numMatrices = NDArrayMath.matricesPerSlice(arr);
        assertEquals(1, numMatrices);
    }

    @Test
    public void testTotalVectors() {
        INDArray arr2 = Nd4j.create(2, 2, 2, 2);
        assertEquals(8, NDArrayMath.numVectors(arr2));
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
