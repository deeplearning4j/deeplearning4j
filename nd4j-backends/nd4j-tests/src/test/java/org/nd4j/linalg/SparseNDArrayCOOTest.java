package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArray;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCSR;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
        double[] values = {1, 2, 3, 4};
        int[][] indices = {{0,3},{1, 2}, {2, 1}, {3, 4}};
        INDArray sparseNDArray = Nd4j.createSparseCOO(values, indices, new int[]{5, 5});

        // subarray in the top right corner
        BaseSparseNDArrayCOO sparseView = (BaseSparseNDArrayCOO) sparseNDArray.get(NDArrayIndex.interval(0,2), NDArrayIndex.interval(0, 2));
        assertEquals(denseView.shapeInfoDataBuffer(), sparseView.shapeInfoDataBuffer());
        double[] currentValues = sparseView.data().asDouble();
        assertArrayEquals(values, currentValues, 1e-5);

        assertArrayEquals(ArrayUtil.flatten(indices), sparseView.getIndices().asInt());
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
        INDArray denseView = array.get(NDArrayIndex.interval(1, 4), NDArrayIndex.point(2));
        System.out.println("dense matrix: \n" + array.toString());

    }

    @Test
    public void shouldTakeView3dimensionArray(){
        double[] data = new double[]{2, 0, 0, 0, 0, 1, 4, 3};
        int[] shape = new int[]{2, 2, 2};
        INDArray array = Nd4j.create(data, shape, 0, 'c');
        System.out.println("GET");
        INDArray denseView = array.get(NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.point(1));
        System.out.println("Shape info full array : " + array.shapeInfoDataBuffer().toString());
        System.out.println("dense matrix: \n" + array.toString());
        System.out.println("Shape info view : " + denseView.shapeInfoDataBuffer().toString());
        System.out.println("dense view: \n" + denseView.toString());
        assert(denseView.isColumnVector());

    }

    @Test
    public void rdm(){
        INDArray arr = Nd4j.rand(new int[]{2, 3, 4, 5});
        INDArray v = arr.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.interval(2,3), NDArrayIndex.interval(1, 3));
        System.out.println(arr);
        System.out.println(v);
    }
}
