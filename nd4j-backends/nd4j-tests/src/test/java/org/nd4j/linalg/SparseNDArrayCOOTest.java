package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;
import org.nd4j.linalg.util.ArrayUtil;

import javax.sound.midi.Soundbank;
import java.sql.DatabaseMetaData;

import static org.junit.Assert.*;

/**
 * @author Audrey Loeffel
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
        assertArrayEquals(shape, sparse.shape());
        assertEquals(data.length, sparse.nnz());

    }

    @Test
    public void shouldPutScalar(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        sparse.putScalar(1, 3);

    }

    @Test
    public void shouldntPutZero(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        int oldNNZ = sparse.nnz();
        sparse.putScalar(1, 0);
        assertArrayEquals(new int[]{0,2}, sparse.getVectorCoordinates().asInt());
        assertTrue(sparse.isRowVector());
        assertEquals(oldNNZ, sparse.nnz());
    }
    @Test
    public void shouldRemoveZero(){
        INDArray sparse = Nd4j.createSparseCOO(new double[]{1,2}, new int[][]{{0, 0},{0, 2}}, new int[]{1, 3});
        sparse.putScalar(0, 0);
        assertArrayEquals(new int[]{2}, sparse.getVectorCoordinates().asInt());
    }

    @Test
    public void shouldTakeViewInLeftTopCorner(){
        // Test with dense ndarray
        double[] data = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0};
        INDArray array = Nd4j.create(data, new int[]{5,5}, 0, 'c');
        INDArray denseView = array.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(0, 2));

        // test with sparse :
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});

        // subarray in the top right corner
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(0, 2));
        assertArrayEquals(denseView.shape(), sparseView.shape());
        double[] currentValues = sparseView.data().asDouble();
        assertArrayEquals(values, currentValues, 1e-5);
        assertArrayEquals(ArrayUtil.flatten(indices), sparseView.getUnderlyingIndices().asInt());
        assertEquals(0, sparseView.nnz());
    }

    @Test
    public void shouldTakeViewInLeftBottomCorner(){

        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});

        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.interval(2,5), NDArrayIndex.interval(0, 2));
        assertEquals(1, sparseView.nnz());
        assertArrayEquals(new double[]{3}, sparseView.getValues().asDouble(), 1e-1);
        assertArrayEquals(new int[]{0,1}, sparseView.getIndices().asInt());
    }

    @Test
    public void shouldTakeViewInRightTopCorner(){

        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(2, 5));
        assertEquals(2, sparseView.nnz());
        assertArrayEquals(new double[]{1, 2}, sparseView.getValues().asDouble(), 1e-1);
        assertArrayEquals(new int[]{0, 1, 1, 0}, sparseView.getIndices().asInt());

    }

    @Test
    public void shouldTakeViewInTheMiddle(){
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.interval(1,3), NDArrayIndex.interval(1, 3));
        assertEquals(2, sparseView.nnz());
        assertArrayEquals(new double[]{2, 3}, sparseView.getValues().asDouble(), 1e-1);
        assertArrayEquals(new int[]{0, 1, 1, 0}, sparseView.getIndices().asInt());

    }
    @Test
    public void shouldGetFirstColumn() {
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.all(), NDArrayIndex.point(0));
        assertEquals(0, sparseView.nnz());
    }

    @Test
    public void shouldGetRowInTheMiddle(){
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.point(2), NDArrayIndex.all());
        assertEquals(1, sparseView.nnz());
        assertArrayEquals(new int[]{0, 1}, sparseView.getIndices().asInt());
        assertArrayEquals(new double[]{3}, sparseView.getValues().asDouble(), 1e-1);

    }

    @Test
    public void shouldGetScalar(){
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.point(2), NDArrayIndex.point(1));
        assertEquals(1, sparseView.nnz());
        assertArrayEquals(new int[]{0, 0}, sparseView.getIndices().asInt());
        assertArrayEquals(new double[]{3}, sparseView.getValues().asDouble(), 1e-1);
        assertTrue(sparseView.isScalar());
    }

    @Test
    public void shouldTakeView3dimensionArray(){
        int[] shape = new int[]{2, 2, 2};
        double[] values = new double[]{2, 1, 4, 3};
        int[][] indices  = new int[][]{{0, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

        INDArray array = Nd4j.createSparseCOO(values, indices, shape);
        BaseSparseNDArrayCOO view = (BaseSparseNDArrayCOO) array.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());
        assertEquals(2, view.nnz());
        assertArrayEquals(new int[]{2, 2}, view.shape());
        assertArrayEquals(new int[]{0, 0, 1, 1}, view.getIndices().asInt());
        assertArrayEquals(new double[]{2, 1}, view.getValues().asDouble(), 1e-1);
    }

    @Test
    public void shouldTakeViewOfView(){
        int[] shape = new int[]{2, 2, 2};
        double[] values = new double[]{2, 1, 4, 3};
        int[][] indices  = new int[][]{{0, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

        INDArray array = Nd4j.createSparseCOO(values, indices, shape);
        BaseSparseNDArrayCOO baseView = (BaseSparseNDArrayCOO) array.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all());
        BaseSparseNDArrayCOO view = (BaseSparseNDArrayCOO) baseView.get(NDArrayIndex.point(1), NDArrayIndex.all());
        assertEquals(1, view.nnz());
        assertArrayEquals(new int[]{1, 2}, view.shape());
        assertArrayEquals(new int[]{0, 1}, view.getIndices().asInt());
        assertArrayEquals(new double[]{1}, view.getValues().asDouble(), 1e-1);
    }

    @Test
    public void shouldTakeViewOfView2(){
        int[] shape = new int[]{4, 2, 3};
        double[] values = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        int[][] indices  = new int[][]{{0, 0, 2}, {0, 1, 1}, {1,0, 0}, {1, 0, 1}, {1, 1, 2},
                {2, 0, 1}, {2, 1, 2}, {3, 0, 1}, {3, 1, 0}};

        INDArray array = Nd4j.createSparseCOO(values, indices, shape);
        BaseSparseNDArrayCOO baseView = (BaseSparseNDArrayCOO) array.get(NDArrayIndex.interval(1, 4), NDArrayIndex.point(1), NDArrayIndex.all());
        BaseSparseNDArrayCOO view = (BaseSparseNDArrayCOO) baseView.get(NDArrayIndex.all(), NDArrayIndex.point(2));
        assertEquals(2, view.nnz());
        assertArrayEquals(new int[]{3, 1}, view.shape());
        assertArrayEquals(new int[]{0, 0, 1, 0}, view.getIndices().asInt());
        assertArrayEquals(new double[]{5, 7}, view.getValues().asDouble(), 1e-1);
    }

    /*
    @Test
    public void rdmTestWithDenseArray(){
        INDArray arr = Nd4j.rand(new int[]{4, 2, 3});
        System.out.println(arr.toString());
        INDArray v = arr.get(NDArrayIndex.interval(1, 4), NDArrayIndex.point(1), NDArrayIndex.all());
        System.out.println("v ");
        System.out.println(v.toString());
        INDArray vv = v.get(NDArrayIndex.all(), NDArrayIndex.point(2));
        System.out.println("vv");
        System.out.println(vv.toString());
        System.out.println(vv.shape()[0] + " "+ vv.shape()[1]);
    }
    */
}
