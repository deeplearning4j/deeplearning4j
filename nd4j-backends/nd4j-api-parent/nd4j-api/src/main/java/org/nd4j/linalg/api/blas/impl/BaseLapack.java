package org.nd4j.linalg.api.blas.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base lapack define float and double versions.
 *
 * @author Adam Gibson
 * @author rcorbish
 */
@Slf4j
public abstract class BaseLapack implements Lapack {

    @Override
    public INDArray getrf(INDArray A) {

        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int m = (int) A.rows();
        int n = (int) A.columns();

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
                        Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}).getFirst());

        int mn = Math.min(m, n);
        INDArray IPIV = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(mn),
                        Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, mn}).getFirst());

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            dgetrf(m, n, A, IPIV, INFO);
        else if (A.data().dataType() == DataBuffer.Type.FLOAT)
            sgetrf(m, n, A, IPIV, INFO);
        else
            throw new UnsupportedOperationException();

        if (INFO.getInt(0) < 0) {
            throw new Error("Parameter #" + INFO.getInt(0) + " to getrf() was not valid");
        } else if (INFO.getInt(0) > 0) {
            log.warn("The matrix is singular - cannot be used for inverse op. Check L matrix at row " + INFO.getInt(0));
        }

        return IPIV;
    }



    /**
    * Float/Double versions of LU decomp.
    * This is the official LAPACK interface (in case you want to call this directly)
    * See getrf for full details on LU Decomp
    *
    * @param M  the number of rows in the matrix A
    * @param N  the number of cols in the matrix A
    * @param A  the matrix to factorize - data must be in column order ( create with 'f' ordering )
    * @param IPIV an output array for the permutations ( must be int based storage )
    * @param INFO error details 1 int array, a positive number (i) implies row i cannot be factored, a negative value implies paramtere i is invalid
    */
    public abstract void sgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO);

    public abstract void dgetrf(int M, int N, INDArray A, INDArray IPIV, INDArray INFO);



    @Override
    public void potrf(INDArray A, boolean lower) {

        // FIXME: int cast
        if (A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        byte uplo = (byte) (lower ? 'L' : 'U'); // upper or lower part of the factor desired ?
        int n = (int) A.columns();

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
                        Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}).getFirst());

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            dpotrf(uplo, n, A, INFO);
        else if (A.data().dataType() == DataBuffer.Type.FLOAT)
            spotrf(uplo, n, A, INFO);
        else
            throw new UnsupportedOperationException();

        if (INFO.getInt(0) < 0) {
            throw new Error("Parameter #" + INFO.getInt(0) + " to potrf() was not valid");
        } else if (INFO.getInt(0) > 0) {
            throw new Error("The matrix is not positive definite! (potrf fails @ order " + INFO.getInt(0) + ")");
        }

        return;
    }



    /**
    * Float/Double versions of cholesky decomp for positive definite matrices    
    * 
    *   A = LL*
    *
    * @param uplo which factor to return L or U 
    * @param A  the matrix to factorize - data must be in column order ( create with 'f' ordering )
    * @param INFO error details 1 int array, a positive number (i) implies row i cannot be factored, a negative value implies paramtere i is invalid
    */
    public abstract void spotrf(byte uplo, int N, INDArray A, INDArray INFO);

    public abstract void dpotrf(byte uplo, int N, INDArray A, INDArray INFO);



    @Override
    public void geqrf(INDArray A, INDArray R) {

        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int m = (int) A.rows();
        int n = (int) A.columns();

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
                        Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}).getFirst());

        if (R.rows() != A.columns() || R.columns() != A.columns()) {
            throw new Error("geqrf: R must be N x N (n = columns in A)");
        }
        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            dgeqrf(m, n, A, R, INFO);
        } else if (A.data().dataType() == DataBuffer.Type.FLOAT) {
            sgeqrf(m, n, A, R, INFO);
        } else {
            throw new UnsupportedOperationException();
        }

        if (INFO.getInt(0) < 0) {
            throw new Error("Parameter #" + INFO.getInt(0) + " to getrf() was not valid");
        }
    }


    /**
    * Float/Double versions of QR decomp.
    * This is the official LAPACK interface (in case you want to call this directly)
    * See geqrf for full details on LU Decomp
    *
    * @param M  the number of rows in the matrix A
    * @param N  the number of cols in the matrix A
    * @param A  the matrix to factorize - data must be in column order ( create with 'f' ordering )
    * @param R  an output array for other part of factorization
    * @param INFO error details 1 int array, a positive number (i) implies row i cannot be factored, a negative value implies paramtere i is invalid
    */
    public abstract void sgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO);

    public abstract void dgeqrf(int M, int N, INDArray A, INDArray R, INDArray INFO);



    @Override
    public int syev(char jobz, char uplo, INDArray A, INDArray V) {

        if (A.rows() != A.columns()) {
            throw new Error("syev: A must be square.");
        }
        if (A.rows() != V.length()) {
            throw new Error("syev: V must be the length of the matrix dimension.");
        }

        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int status = -1;
        if (A.data().dataType() == DataBuffer.Type.DOUBLE) {
            status = dsyev(jobz, uplo, (int) A.rows(), A, V);
        } else if (A.data().dataType() == DataBuffer.Type.FLOAT) {
            status = ssyev(jobz, uplo, (int) A.rows(), A, V);
        } else {
            throw new UnsupportedOperationException();
        }

        return status;
    }


    /**
    * Float/Double versions of eigen value/vector calc.
    *
    * @param jobz 'N' - no eigen vectors, 'V' - return eigenvectors
    * @param uplo upper or lower part of symmetric matrix to use
    * @param N  the number of rows & cols in the matrix A
    * @param A  the matrix to calculate eigenvectors
    * @param R  an output array for eigenvalues ( may be null )
    */
    public abstract int ssyev(char jobz, char uplo, int N, INDArray A, INDArray R);

    public abstract int dsyev(char jobz, char uplo, int N, INDArray A, INDArray R);



    @Override
    public void gesvd(INDArray A, INDArray S, INDArray U, INDArray VT) {
        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int m = (int) A.rows();
        int n = (int) A.columns();

        byte jobu = (byte) (U == null ? 'N' : 'A');
        byte jobvt = (byte) (VT == null ? 'N' : 'A');

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1),
                        Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {1, 1}).getFirst());

        if (A.data().dataType() == DataBuffer.Type.DOUBLE)
            dgesvd(jobu, jobvt, m, n, A, S, U, VT, INFO);
        else if (A.data().dataType() == DataBuffer.Type.FLOAT)
            sgesvd(jobu, jobvt, m, n, A, S, U, VT, INFO);
        else
            throw new UnsupportedOperationException();

        if (INFO.getInt(0) < 0) {
            throw new Error("Parameter #" + INFO.getInt(0) + " to gesvd() was not valid");
        } else if (INFO.getInt(0) > 0) {
            log.warn("The matrix contains singular elements. Check S matrix at row " + INFO.getInt(0));
        }
    }

    public abstract void sgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO);

    public abstract void dgesvd(byte jobu, byte jobvt, int M, int N, INDArray A, INDArray S, INDArray U, INDArray VT,
                    INDArray INFO);



    @Override
    public INDArray getPFactor(int M, INDArray ipiv) {
        // The simplest permutation is the identity matrix
        INDArray P = Nd4j.eye(M); // result is a square matrix with given size
        for (int i = 0; i < ipiv.length(); i++) {
            int pivot = ipiv.getInt(i) - 1; // Did we swap row #i with anything?
            if (pivot > i) { // don't reswap when we get lower down in the vector
                INDArray v1 = P.getColumn(i).dup(); // because of row vs col major order we'll ...
                INDArray v2 = P.getColumn(pivot); // ... make a transposed matrix immediately
                P.putColumn(i, v2);
                P.putColumn(pivot, v1); // note dup() above is required - getColumn() is a 'view'
            }
        }
        return P; // the permutation matrix - contains a single 1 in any row and column
    }


    /* TODO: consider doing this in place to save memory. This implies U is taken out first
       L is the same shape as the input matrix. Just the lower triangular with a diagonal of 1s
     */
    @Override
    public INDArray getLFactor(INDArray A) {
        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int m = (int) A.rows();
        int n = (int) A.columns();

        INDArray L = Nd4j.create(m, n);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (r > c && r < m && c < n) {
                    L.putScalar(r, c, A.getFloat(r, c));
                } else if (r < c) {
                    L.putScalar(r, c, 0.f);
                } else {
                    L.putScalar(r, c, 1.f);
                }
            }
        }
        return L;
    }


    @Override
    public INDArray getUFactor(INDArray A) {
        // FIXME: int cast
        if (A.rows() > Integer.MAX_VALUE || A.columns() > Integer.MAX_VALUE)
            throw new ND4JArraySizeException();

        int m = (int) A.rows();
        int n = (int) A.columns();

        INDArray U = Nd4j.create(n, n);

        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                if (r <= c && r < m && c < n) {
                    U.putScalar(r, c, A.getFloat(r, c));
                } else {
                    U.putScalar(r, c, 0.f);
                }
            }
        }
        return U;
    }

}
