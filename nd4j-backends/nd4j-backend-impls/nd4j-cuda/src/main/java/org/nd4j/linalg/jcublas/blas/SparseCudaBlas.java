package org.nd4j.linalg.jcublas.blas;


import org.nd4j.linalg.api.blas.Blas;
import org.nd4j.nativeblas.SparseNd4jBlas;

/**
 * @author Audrey Loeffel
 */
public class SparseCudaBlas extends SparseNd4jBlas {
    @Override
    public void setMaxThreads(int num){

    }

    @Override
    public int getMaxThreads() {
        return 0;
    }

    @Override
    public int getBlasVendorId() {
        return 0;
    }
}
