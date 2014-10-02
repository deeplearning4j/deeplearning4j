package org.nd4j.linalg.eigen;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Compute eigen values
 *
 * @author Adam Gibson
 */
public class Eigen {

    public static INDArray dummy = Nd4j.scalar(1);

    /**
     * Computes the eigenvalues of a general matrix.
     */
    public static IComplexNDArray eigenvalues(INDArray A) {
        assert A.rows() == A.columns();
        INDArray WR = Nd4j.create(A.rows());
        INDArray WI = WR.dup();
        Nd4j.getBlasWrapper().geev(
                'N',
                'N',
                A.dup(),
                WR,
                WI,
                dummy,
                dummy);
        return Nd4j.createComplex(WR, WI);
    }


    /**
     * Compute generalized eigenvalues of the problem A x = L B x.
     *
     * @param A symmetric Matrix A. Only the upper triangle will be considered.
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A) {
        INDArray eigenvalues = Nd4j.create(A.rows());
        int isuppz[] = new int[2 * A.rows()];
        Nd4j.getBlasWrapper().syevr(
                'N',
                'A',
                'U',
                A.dup(),
                0,
                0,
                0,
                0,
                0,
                eigenvalues,
                Nd4j.ones(1),
                isuppz);
        return eigenvalues;

    }



    /**
     * Computes the eigenvalues and eigenvectors of a general matrix.
     * @param A the ndarray to get the eigen vectors for
     * @return 2 arrays representing W (eigen vectors) and V (normalized eigen vectors)
     */
    public static IComplexNDArray[] eigenvectors(INDArray A) {
        assert A.columns() == A.rows();
        // setting up result arrays
        INDArray WR = Nd4j.create(A.rows());
        INDArray WI = WR.dup();
        INDArray VR = Nd4j.create(A.rows(), A.rows());
        INDArray VL = Nd4j.create(A.rows());

        Nd4j.getBlasWrapper().geev('v','v',A.dup(),WR,WI,VL,VR);

        // transferring the result
        IComplexNDArray E = Nd4j.createComplex(WR, WI);
        IComplexNDArray V =  Nd4j.createComplex(A.rows(), A.rows());
        //System.err.printf("VR = %s\n", VR.toString());
        for (int i = 0; i < A.rows(); i++) {
            if (E.getComplex(i).isReal()) {
                IComplexNDArray column = Nd4j.createComplex(VR.getColumn(i));
                V.putColumn(i,column);
            }
            else {
                IComplexNDArray v = Nd4j.createComplex(VR.getColumn(i), VR.getColumn(i + 1));
                V.putColumn(i, v);
                V.putColumn(i + 1, v.conji());
                i += 1;
            }
        }
        return new IComplexNDArray[]{ Nd4j.diag(E),V};
    }




    /**
     * Compute generalized eigenvalues of the problem A x = L B x.
     *
     * @param A symmetric Matrix A. Only the upper triangle will be considered.
     * @param B symmetric Matrix B. Only the upper triangle will be considered.
     * @return a vector of eigenvalues L.
     */
    public static INDArray symmetricGeneralizedEigenvalues(INDArray A, INDArray B) {
        assert A.rows() == A.columns();
        assert B.rows() == B.columns();
        INDArray W = Nd4j.create(A.rows());
        Nd4j.getBlasWrapper().sygvd(1, 'N', 'U', A.dup(), B.dup(), W);
        return W;
    }


}
