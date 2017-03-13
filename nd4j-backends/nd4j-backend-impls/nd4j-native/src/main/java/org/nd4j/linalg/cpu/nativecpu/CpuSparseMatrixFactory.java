package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.cpu.nativecpu.blas.*;
import org.nd4j.linalg.factory.ISparseMatrixFactory;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Audrey Loeffel
 */
public class CpuSparseMatrixFactory extends ISparseMatrixFactory {

    public CpuSparseMatrixFactory(){}

    static{
        Nd4j.getBlasWrapper();
    }
    // contructors ?

    Override
    public void createBlas(){ blas = new SparseCpuBlas();}

    @Override
    public void createLevel1() {
        level1 = new SparseCpuLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new SparseCpuLevel2();
    }

    @Override
    public void createLevel3() {

        level3 = new SparseCpuLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new SparseCpuLapack();
    }
}
