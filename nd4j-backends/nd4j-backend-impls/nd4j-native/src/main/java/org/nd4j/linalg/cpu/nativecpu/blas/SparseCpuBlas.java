package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.nativeblas.Nd4jBlas;

/**
 * Created by audrey on 3/13/17.
 */
public class SparseCpuBlas extends Nd4jBlas {
    // TODO : conversion char parameters <-> MKL constants
    // TODO : max threads
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
