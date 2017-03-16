package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel2;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.Nd4jBlas;

/**
 * @author Audrey Loeffel
 */
public class SparseCpuLevel2 extends SparseBaseLevel2 {
    private Nd4jBlas nd4jBlas = (Nd4jBlas) Nd4j.factory().blas();
    // Mapping with Sparse Blas calls
}
