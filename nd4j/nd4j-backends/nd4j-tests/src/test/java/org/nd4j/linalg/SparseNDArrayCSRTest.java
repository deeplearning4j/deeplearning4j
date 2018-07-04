package org.nd4j.linalg;

import org.junit.Ignore;
import org.junit.Test;

import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCSR;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.*;


/**
 * @author Audrey Loeffel
 */
@Ignore // temporary ignored
public class SparseNDArrayCSRTest {

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
    private long[] shape = {5, 5};


    @Test
    public void shouldCreateSparseMatrix() {
        INDArray matrix = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        //TODO
    }

    @Test
    public void shouldAddValueAtAGivenPosition() {
        /*
        * [[1 -1 0 -3 0]
        *  [-2 4 0 0 0 ]
        *  [ 0 3 4 6 4 ] = A'
        *  [-4 0 2 7 0 ]
        *  [ 0 8 0 0 -5]]
        * */
        INDArray sparseNDArray = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        if (sparseNDArray instanceof BaseSparseNDArrayCSR) {
            BaseSparseNDArrayCSR sparseCSRArray = (BaseSparseNDArrayCSR) sparseNDArray;
            sparseCSRArray.putScalar(2, 1, 3);

            double[] expectedValues = {1, -2, -3, -2, 5, 3, 4, 6, 4, -4, 2, 7, 8, -5};
            double[] expectedColumns = {0, 1, 3, 0, 1, 1, 2, 3, 4, 0, 2, 3, 1, 4};
            int[] expectedPointerB = {0, 3, 5, 9, 12};
            int[] expectedPointerE = {3, 5, 9, 12, 14};
            long[] expectedShape = {5, 5};


            assertArrayEquals(expectedValues, sparseCSRArray.getDoubleValues(), 0);
            assertArrayEquals(expectedColumns, sparseCSRArray.getColumns(), 0);
            assertArrayEquals(expectedPointerB, sparseCSRArray.getPointerBArray());
            assertArrayEquals(expectedPointerE, sparseCSRArray.getPointerEArray());
            assertArrayEquals(expectedShape, shape);
        }
    }

    @Test
    public void shouldReallocate() {
        INDArray sparseNDArray = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        if (sparseNDArray instanceof BaseSparseNDArrayCSR) {
            BaseSparseNDArrayCSR sparseCSRArray = (BaseSparseNDArrayCSR) sparseNDArray;
            int initialSize = sparseCSRArray.getDoubleValues().length;

            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    sparseCSRArray.putScalar(i, j, i + j);
                }
            }
            int finalSize = sparseCSRArray.getDoubleValues().length;
            assert (finalSize > initialSize);
        }

    }

    @Test
    public void shouldReplaceValueAtAGivenPosition() {
        /*
        * [[1 -1 0 -3 0]
        *  [-2 4 0 0 0 ]
        *  [ 0 0 10 6 4] = A'
        *  [-4 0 2 7 0 ]
        *  [ 0 8 0 0 -5]]
        * */
        INDArray sparseNDArray = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        if (sparseNDArray instanceof BaseSparseNDArrayCSR) {
            BaseSparseNDArrayCSR sparseCSRArray = (BaseSparseNDArrayCSR) sparseNDArray;
            sparseCSRArray.putScalar(2, 2, 10);

            double[] expectedValues = {1, -2, -3, -2, 5, 10, 6, 4, -4, 2, 7, 8, -5};
            double[] expectedColumns = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
            int[] expectedPointerB = {0, 3, 5, 8, 11};
            int[] expectedPointerE = {3, 5, 8, 11, 13};
            long[] expectedShape = {5, 5};

            assertArrayEquals(expectedValues, sparseCSRArray.getDoubleValues(), 0);
            assertArrayEquals(expectedColumns, sparseCSRArray.getColumns(), 0);
            assertArrayEquals(expectedPointerB, sparseCSRArray.getPointerBArray());
            assertArrayEquals(expectedPointerE, sparseCSRArray.getPointerEArray());
            assertArrayEquals(expectedShape, shape);
        }
    }

    @Test
    public void shouldGetValueAtAGivenPosition() {
        // Not yet implemented
    }

    @Test
    public void shouldBeEqualToDense() {
        // Not yet implemented
    }

    @Test
    public void shouldGetAView() {

        double[] values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, 5};
        int[] columns = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
        int[] pointerB = {0, 3, 5, 8, 11};
        int[] pointerE = {3, 5, 8, 11, 13};

        // Test with dense ndarray
        double[] data = {1, -1, 0, -3, 0, -2, 5, 0, 0, 0, 0, 0, 4, 6, 4, -4, 0, 2, 7, 0, 0, 8, 0, 0, 5};
        INDArray array = Nd4j.create(data, new int[] {5, 5}, 0, 'c');
        INDArray denseView = array.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3));

        // test with sparse :
        INDArray sparseNDArray = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);

        // subarray in the top right corner
        BaseSparseNDArrayCSR sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.interval(0, 3),
                        NDArrayIndex.interval(3, 5));
        assertArrayEquals(new int[] {0, 0, 1}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {2, 3, 6}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {3, 3, 8}, sparseView.getPointerEArray());

        // subarray in the middle
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3));
        assertArrayEquals(new int[] {0, 1}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {4, 5}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {5, 6}, sparseView.getPointerEArray());

        // get the first row
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertArrayEquals(new int[] {0, 0, 0}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 3, 4, 8, 9}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {1, 4, 4, 9, 9}, sparseView.getPointerEArray());

        // get several rows
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.interval(0, 2), NDArrayIndex.all());
        assertArrayEquals(new int[] {0, 1, 3, 0, 1}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 3}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {3, 5}, sparseView.getPointerEArray());

        // get a row in the middle
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.point(2), NDArrayIndex.all());
        assertArrayEquals(new int[] {2, 3, 4}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {5}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {8}, sparseView.getPointerEArray());

        // get the first column
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertArrayEquals(new int[] {0, 0, 0}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 3, 4, 8, 9}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {1, 4, 4, 9, 9}, sparseView.getPointerEArray());

        // get a column in the middle
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.all(), NDArrayIndex.point(2));
        assertArrayEquals(new int[] {0, 0}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 0, 5, 9, 10}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {0, 0, 6, 10, 10}, sparseView.getPointerEArray());

        // get a part of the column in the middle
        sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.interval(1, 4), NDArrayIndex.point(2));
        assertArrayEquals(new int[] {0, 0}, sparseView.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 5, 9}, sparseView.getPointerBArray());
        assertArrayEquals(new int[] {0, 6, 10}, sparseView.getPointerEArray());


    }

    @Test
    public void shouldGetAViewFromView() {
        double[] values = {1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, 5};
        int[] columns = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
        int[] pointerB = {0, 3, 5, 8, 11};

        INDArray sparseNDArray = Nd4j.createSparseCSR(values, columns, pointerB, pointerE, shape);
        /*             [0, -3, 0]
        * sparseView = [0,  0, 0] subview = [[0,0], [4,6]]
        *              [4,  6, 4]
        */
        BaseSparseNDArrayCSR sparseView = (BaseSparseNDArrayCSR) sparseNDArray.get(NDArrayIndex.interval(0, 3),
                        NDArrayIndex.interval(2, 5));
        BaseSparseNDArrayCSR subview =
                        (BaseSparseNDArrayCSR) sparseView.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(0, 2));
        assertArrayEquals(new int[] {0, 1}, subview.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 5}, subview.getPointerBArray());
        assertArrayEquals(new int[] {0, 7}, subview.getPointerEArray());

        // get the first column
        subview = (BaseSparseNDArrayCSR) sparseView.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertArrayEquals(new int[] {0}, subview.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0, 0, 5}, subview.getPointerBArray());
        assertArrayEquals(new int[] {0, 0, 6}, subview.getPointerEArray());

        // get a column in the middle
        subview = (BaseSparseNDArrayCSR) sparseView.get(NDArrayIndex.all(), NDArrayIndex.point(1));
        assertArrayEquals(new int[] {0, 0}, subview.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {2, 3, 6}, subview.getPointerBArray());
        assertArrayEquals(new int[] {3, 3, 7}, subview.getPointerEArray());

        // get the first row
        subview = (BaseSparseNDArrayCSR) sparseView.get(NDArrayIndex.point(0), NDArrayIndex.all());
        assertArrayEquals(new int[] {1}, subview.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {2}, subview.getPointerBArray());
        assertArrayEquals(new int[] {3}, subview.getPointerEArray());

        // get a row in the middle
        subview = (BaseSparseNDArrayCSR) sparseView.get(NDArrayIndex.point(1), NDArrayIndex.all());
        assertArrayEquals(new int[] {}, subview.getVectorCoordinates().asInt());
        assertArrayEquals(new int[] {0}, subview.getPointerBArray());
        assertArrayEquals(new int[] {0}, subview.getPointerEArray());
    }
}
