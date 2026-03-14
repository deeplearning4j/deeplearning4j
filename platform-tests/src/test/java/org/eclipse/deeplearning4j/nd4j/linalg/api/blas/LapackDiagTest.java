package org.eclipse.deeplearning4j.nd4j.linalg.api.blas;

import org.bytedeco.javacpp.DoublePointer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LapackDiagTest {
    @Test
    public void diagCholesky() {
        // Test with f-order directly using raw LAPACKE call
        System.out.println("=== Direct LAPACKE call with COL_MAJOR + 'L' ===");
        double[] rawF = {2, -1, 1, -1, 2, -1, 1, -1, 2};
        DoublePointer ptrF = new DoublePointer(rawF);
        int status = Nd4j.getBlasLapackDelegator().LAPACKE_dpotrf(102, (byte)'L', 3, ptrF, 3);
        System.out.println("Status: " + status);
        System.out.print("Raw buffer after COL_MAJOR+L: ");
        for (int i = 0; i < 9; i++) System.out.printf("%.4f ", ptrF.get(i));
        System.out.println();

        System.out.println("\n=== Direct LAPACKE call with ROW_MAJOR + 'L' ===");
        double[] rawR = {2, -1, 1, -1, 2, -1, 1, -1, 2};
        DoublePointer ptrR = new DoublePointer(rawR);
        status = Nd4j.getBlasLapackDelegator().LAPACKE_dpotrf(101, (byte)'L', 3, ptrR, 3);
        System.out.println("Status: " + status);
        System.out.print("Raw buffer after ROW_MAJOR+L: ");
        for (int i = 0; i < 9; i++) System.out.printf("%.4f ", ptrR.get(i));
        System.out.println();

        System.out.println("\n=== NDArray test with c-order, potrf(lower=true) ===");
        INDArray A = Nd4j.create(new double[] {2, -1, 1, -1, 2, -1, 1, -1, 2});
        A = A.reshape('c', 3, 3);
        System.out.println("A original:\n" + A);
        INDArray O = A.dup();
        Nd4j.getBlasWrapper().lapack().potrf(A, true);
        System.out.println("A after potrf(lower=true):\n" + A);
        System.out.print("A raw buffer: ");
        for (int i = 0; i < 9; i++) System.out.printf("%.4f ", A.data().getDouble(i));
        System.out.println();
        INDArray product = A.mmul(A.transpose());
        System.out.println("L * L^T:\n" + product);
        System.out.println("Diff:\n" + O.sub(product));
    }
}
