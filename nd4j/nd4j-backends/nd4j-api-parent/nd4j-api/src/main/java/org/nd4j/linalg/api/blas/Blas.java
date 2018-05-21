package org.nd4j.linalg.api.blas;

/**
 * Extra functionality for BLAS
 *
 * @author saudet
 */
public interface Blas {

    public enum Vendor {

        UNKNOWN, CUBLAS, OPENBLAS, MKL,
    }

    void setMaxThreads(int num);

    int getMaxThreads();

    /**
     * Returns the BLAS library vendor id
     *
     * 0 - UNKNOWN, 1 - CUBLAS, 2 - OPENBLAS, 3 - MKL
     *
     * @return the BLAS library vendor id
     */
    int getBlasVendorId();

    /**
     * Returns the BLAS library vendor
     *
     * @return the BLAS library vendor
     */
    public Vendor getBlasVendor();
}
