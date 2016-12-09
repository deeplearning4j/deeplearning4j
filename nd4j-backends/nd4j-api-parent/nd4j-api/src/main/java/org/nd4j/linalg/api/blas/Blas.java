package org.nd4j.linalg.api.blas;

/**
 * Extra functionality for BLAS
 *
 * @author saudet
 */
public interface Blas {
    void setMaxThreads(int num);

    int getMaxThreads();

    int getVendor();
}
