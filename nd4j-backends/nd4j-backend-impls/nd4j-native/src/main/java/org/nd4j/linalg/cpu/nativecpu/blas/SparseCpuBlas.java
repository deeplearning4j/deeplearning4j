package org.nd4j.linalg.cpu.nativecpu.blas;

import org.nd4j.nativeblas.SparseNd4jBlas;


import static org.bytedeco.javacpp.mkl_rt.*;
/**
 * @author Audrey Loeffel
 */
public class SparseCpuBlas extends SparseNd4jBlas {
    // TODO : conversion char parameters <-> MKL constants

    @Override
    public void setMaxThreads(int num){
        MKL_Set_Num_Threads(num);
    }

    @Override
    public int getMaxThreads() {
        return MKL_Get_Max_Threads();
    }

    @Override
    public int getBlasVendorId() {

        //TODO What is that ?

        return 0;
    }
}
