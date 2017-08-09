package org.nd4j.nativeblas;

import org.nd4j.linalg.api.blas.Blas;

public abstract class SparseNd4jBlas implements Blas {

    public SparseNd4jBlas() {

    }

    /**
     * Returns the BLAS library vendor
     *
     * @return the BLAS library vendor
     */
    @Override
    public Vendor getBlasVendor() {
        int vendor = getBlasVendorId();
        boolean isUnknowVendor = ((vendor > Vendor.values().length - 1) || (vendor <= 0));
        if (isUnknowVendor) {
            return Vendor.UNKNOWN;
        }
        return Vendor.values()[vendor];
    }
}
