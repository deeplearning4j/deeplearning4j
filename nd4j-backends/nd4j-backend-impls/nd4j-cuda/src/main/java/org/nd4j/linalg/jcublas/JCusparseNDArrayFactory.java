package org.nd4j.linalg.jcublas;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.ISparseNDArray;
import org.nd4j.linalg.factory.BaseSparseNDArrayFactory;
import org.nd4j.linalg.jcublas.blas.*;
import org.nd4j.nativeblas.NativeOps;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public class JCusparseNDArrayFactory extends BaseSparseNDArrayFactory{

    private NativeOps nativeOps = null ; //TODO

    public JCusparseNDArrayFactory(){}

    @Override
    public void createBlas() {
        blas = new SparseCudaBlas();
    }

    @Override
    public void createLevel1() {
        level1 = new JcusparseLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new JcusparseLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new JcusparseLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new JcusparseLapack();
    }

    @Override
    public ISparseNDArray createSparse(double[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {
        return new JcusparseNDArrayCSR(data, columns, pointerB, pointerE, shape);
    }
}
