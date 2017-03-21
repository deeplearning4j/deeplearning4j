package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.blas.*;

/**
 * @author Audrey Loeffel
 */
public abstract class BaseISparseMatrixFactory implements ISparseMatrixFactory{
    protected char order; // c or f
    protected Blas blas;
    protected Level1 level1;
    protected Level2 level2;
    protected Level3 level3;
    protected Lapack lapack;

    @Override
    public Lapack lapack() {
        if (lapack == null)
            createLapack();
        return lapack;
    }

    @Override
    public Blas blas() {
        if (blas == null)
            createBlas();
        return blas;
    }

    @Override
    public Level1 level1() {
        if (level1 == null)
            createLevel1();
        return level1;
    }

    @Override
    public Level2 level2() {
        if (level2 == null)
            createLevel2();
        return level2;
    }

    @Override
    public Level3 level3() {
        if (level3 == null)
            createLevel3();
        return level3;
    }
}
