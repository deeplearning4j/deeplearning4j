package org.nd4j.linalg.jcublas.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel2;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class JcusparseLevel2 extends SparseBaseLevel2 {


    @Override
    protected void scoomv(char transA, int M, DataBuffer values, DataBuffer rowInd, DataBuffer colInd, int nnz, INDArray x, INDArray y) {

    }

    @Override
    protected void dcoomv(char transA, int M, DataBuffer values, DataBuffer rowInd, DataBuffer colInd, int nnz, INDArray x, INDArray y) {

    }
}
