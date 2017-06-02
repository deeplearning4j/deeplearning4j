package org.nd4j.linalg;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
}
