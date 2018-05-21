package org.nd4j.nativeblas;


import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.blas.Blas;


/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 */
@Slf4j
public abstract class Nd4jBlas implements Blas {


    public Nd4jBlas() {
        int numThreads;
        String skipper = System.getenv("ND4J_SKIP_BLAS_THREADS");
        if (skipper == null || skipper.isEmpty()) {
            String numThreadsString = System.getenv("OMP_NUM_THREADS");
            if (numThreadsString != null && !numThreadsString.isEmpty()) {
                numThreads = Integer.parseInt(numThreadsString);
                setMaxThreads(numThreads);
            } else {
                int cores = Loader.totalCores();
                int chips = Loader.totalChips();
                if (cores > 0 && chips > 0)
                    numThreads = Math.max(1, cores / chips);
                else
                    numThreads = NativeOps.getCores(Runtime.getRuntime().availableProcessors());
                setMaxThreads(numThreads);
            }
            log.info("Number of threads used for BLAS: {}", getMaxThreads());
        }
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
