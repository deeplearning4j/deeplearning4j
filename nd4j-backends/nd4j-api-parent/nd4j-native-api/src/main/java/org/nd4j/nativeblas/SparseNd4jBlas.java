package org.nd4j.nativeblas;

import org.nd4j.linalg.api.blas.Blas;

public abstract class SparseNd4jBlas implements Blas {

    public SparseNd4jBlas(){

    }

    @Override
    public void setMaxThreads(int num) {

    }

    @Override
    public int getMaxThreads() {
        return 0;
    }

    @Override
    public int getBlasVendorId() {
        return 0;
    }

    @Override
    public Vendor getBlasVendor() {
        return null;
    }
}
