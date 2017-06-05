package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCSR;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.sql.DatabaseMetaData;

import static org.junit.Assert.*;

/**
 * @author
 */
public class SparseNDArrayCOOTest {

    double[] data = {10,1,2,3,4,5};
    int[] shape = {2,2,2};
    int[][] indices = new int[][]{
            new int[]{0, 0, 0, 1, 2, 2},
            new int[]{0, 0, 1, 1, 1, 2},
            new int[]{1, 2, 2, 1, 0, 1}};


    @Test
    public void shouldCreateSparseMatrix() {
        INDArray sparse = Nd4j.createSparseCOO(data, indices, shape);
        //TODO
    }

    @Test
    public void shouldPutScalar(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        sparse.putScalar(1, 3);

    }

    @Test
    public void shouldntPutZero(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        sparse.putScalar(1, 0);
        BaseSparseNDArrayCOO coo =(BaseSparseNDArrayCOO) sparse;
        int[] coord = coo.getVectorCoordinates().asInt();
        assertArrayEquals(new int[]{0,2}, coo.getVectorCoordinates().asInt());
    }
    @Test
    public void shouldRemoveZero(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        sparse.putScalar(0, 0);
        BaseSparseNDArrayCOO coo =(BaseSparseNDArrayCOO) sparse;
        int[] coord = coo.getVectorCoordinates().asInt();
        assertArrayEquals(new int[]{2}, coo.getVectorCoordinates().asInt());
    }

    // TODO - get view tests

    @Test
    public void shouldTakeViewInLeftTopCorner(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(0, 2));

        // test with sparse :
        INDArray sparseNDArray;

        // subarray in the top right corner
        BaseSparseNDArrayCSR sparseView ;
    }

    @Test
    public void shouldTakeViewInLeftBottomCorner(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.interval(3,5), NDArrayIndex.interval(0, 2));

       }

    @Test
    public void shouldTakeViewInRightTopCorner(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(3, 5));

    }

    @Test
    public void shouldTakeViewInTheMiddle(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.interval(1,3), NDArrayIndex.interval(1, 4));

    }
    @Test
    public void shouldGetFirstColumn() {
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5, 5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.all(), NDArrayIndex.point(0));

    }

    @Test
    public void shouldGetRowInTheMiddle(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.point(2), NDArrayIndex.all());

    }

    @Test
    public void shouldGetPartOfColumn(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        System.out.println("dense matrix: \n" + array.toString());
        INDArray denseView = array.get(NDArrayIndex.interval(1, 4), NDArrayIndex.point(2));

    }
}
