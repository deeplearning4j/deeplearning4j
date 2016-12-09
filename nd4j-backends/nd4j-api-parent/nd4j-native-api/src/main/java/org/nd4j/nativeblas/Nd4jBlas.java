package org.nd4j.nativeblas;


import org.nd4j.linalg.api.blas.Blas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CBlas bindings
 *
 * Original credit:
 * https://github.com/uncomplicate/neanderthal-atlas
 */
public abstract class Nd4jBlas implements Blas {

    public enum Vendor {
        UNKNOWN,
        CUBLAS,
        OPENBLAS,
        MKL,
    }


    private static Logger logger = LoggerFactory.getLogger(Nd4jBlas.class);

    public Nd4jBlas() {
        int numThreads;
        String skipper = System.getenv("ND4J_SKIP_BLAS_THREADS");
        if (skipper == null || skipper.isEmpty()) {
            String numThreadsString = System.getenv("OMP_NUM_THREADS");
            if (numThreadsString != null && !numThreadsString.isEmpty()) {
                numThreads = Integer.parseInt(numThreadsString);
                setMaxThreads(numThreads);
            } else {
                numThreads = getCores(Runtime.getRuntime().availableProcessors());
                setMaxThreads(numThreads);
            }
            logger.info("Number of threads used for BLAS: {}", getMaxThreads());
        }
    }

    private int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256) return  64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4) return 4; // special case for Intel i5. and nobody likes i3 anyway

        if (ht_off > 24) {
            int rounds = 0;
            while (ht_off > 24) { // we loop until final value gets below 24 cores, since that's reasonable threshold as of 2016
                if (ht_off > 24) {
                    ht_off /= 2; // we dont' have any cpus that has higher number then 24 physical cores
                    rounds++;
                }
            }
            // 20 threads is special case in this branch
            if (ht_off == 20 && rounds < 2)
                ht_off /= 2;
        } else { // low-core models are known, but there's a gap, between consumer cpus and xeons
            if (ht_off <= 6) {
                // that's more likely consumer-grade cpu, so leave this value alone
                return ht_off;
            } else {
                if (isOdd(ht_off)) // if that's odd number, it's final result
                    return ht_off;

                // 20 threads & 16 threads are special case in this branch, where we go min value
                if (ht_off == 20 || ht_off == 16)
                    ht_off /= 2;
            }
        }
        return ht_off;
    }

    /**
     * This method returns BLAS library vendor
     *
     * @return
     */
    public Vendor getBlasVendor() {
        if (getVendor() > 3)
            return Vendor.UNKNOWN;

        return Vendor.values()[getVendor()];
    }

    private boolean isOdd(int value) {
        return (value % 2 != 0);
    }
}
