package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel1;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.SparseNd4jBlas;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.

import static org.bytedeco.javacpp.mkl_rt.*;

/**
 * @author Audrey Loeffels
 */
public class SparseCpuLevel1 extends SparseBaseLevel1 {

    private SparseNd4jBlas sparseNd4jBlas = (SparseNd4jBlas) Nd4j.sparseFactory().blas();
    // Mapping with Sparse Blas calls

}
