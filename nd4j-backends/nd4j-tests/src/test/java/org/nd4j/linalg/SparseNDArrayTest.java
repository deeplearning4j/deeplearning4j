package org.nd4j.linalg;

import org.junit.Test;
import static org.junit.Assert.assertArrayEquals;
import org.nd4j.linalg.api.ndarray.ISparseNDArray;
import org.nd4j.linalg.cpu.nativecpu.CpuCSRSparseNDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Audrey Loeffel
 */
public class SparseNDArrayTest {

    /*
    * [[1 -1 0 -3 0]
    *  [-2 4 0 0 0 ]
    *  [ 0 0 4 6 4 ] = A
    *  [-4 0 2 7 0 ]
    *  [ 0 8 0 0 -5]]
    * */

    // CSR representation of the matrix A according to https://software.intel.com/en-us/node/599835
    private double[] values = {1, -2, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5};
    private int[] columns = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    private int[] pointerB = {0, 3, 5, 8, 11};
    private int[] pointerE = {3, 5, 8, 11, 13};
    private int[] shape = {5, 5};


    @Test
    public void shouldCreateSparseMatrix() {
        ISparseNDArray matrix = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        //TODO
    }

    @Test
    public void shouldAddValueAtAGivenPosition(){
        /*
        * [[1 -1 0 -3 0]
        *  [-2 4 0 0 0 ]
        *  [ 0 3 4 6 4 ] = A'
        *  [-4 0 2 7 0 ]
        *  [ 0 8 0 0 -5]]
        * */
        ISparseNDArray matrix = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        if(matrix instanceof CpuCSRSparseNDArray) {
            CpuCSRSparseNDArray csrMatrix = (CpuCSRSparseNDArray) matrix;
            csrMatrix.putScalar(2, 1, 3);

            double[] expectedValues = {1, -2, -3, -2, 5, 3, 4, 6, 4, -4, 2, 7, 8, -5};
            int[] expectedColumns = {0, 1, 3, 0, 1, 1, 2, 3, 4, 0, 2, 3, 1, 4};
            int[] expectedPointerB = {0, 3, 5, 9, 12};
            int[] expectedPointerE = {3, 5, 9, 12, 14};
            int[] expectedShape = {5, 5};


            assertArrayEquals(expectedValues, values, 0);
            assertArrayEquals(expectedColumns, columns);
            assertArrayEquals(expectedPointerB, pointerB);
            assertArrayEquals(expectedPointerE, pointerE);
            assertArrayEquals(expectedShape, shape);
        }
    }

    @Test
    public void shouldReplaceValueAtAGivenPosition(){
        /*
        * [[1 -1 0 -3 0]
        *  [-2 4 0 0 0 ]
        *  [ 0 0 10 6 4] = A'
        *  [-4 0 2 7 0 ]
        *  [ 0 8 0 0 -5]]
        * */
        ISparseNDArray matrix = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        if(matrix instanceof CpuCSRSparseNDArray) {
            CpuCSRSparseNDArray csrMatrix = (CpuCSRSparseNDArray) matrix;
            csrMatrix.putScalar(2, 2, 10);

            double[] expectedValues = {1, -2, -3, -2, 5, 10, 6, 4, -4, 2, 7, 8, -5};
            int[] expectedColumns = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
            int[] expectedPointerB = {0, 3, 5, 8, 11};
            int[] expectedPointerE = {3, 5, 8, 11, 13};
            int[] expectedShape = {5, 5};

            assertArrayEquals(expectedValues, values, 0);
            assertArrayEquals(expectedColumns, columns);
            assertArrayEquals(expectedPointerB, pointerB);
            assertArrayEquals(expectedPointerE, pointerE);
            assertArrayEquals(expectedShape, shape);
        }
    }

    @Test
    public void shouldGetValueAtAGivenPosition(){
        // Not yet implemented
    }

    @Test
    public void shouldBeEqualToDense(){
        // Not yet implemented
    }
}
