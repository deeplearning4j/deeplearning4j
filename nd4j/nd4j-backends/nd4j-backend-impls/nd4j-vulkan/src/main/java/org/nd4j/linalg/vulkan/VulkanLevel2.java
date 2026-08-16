/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.api.blas.Level2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.TriangularSolve;
import org.nd4j.linalg.api.ops.custom.Triu;
import org.nd4j.linalg.factory.Nd4j;

/** Device-backed Level-2 BLAS operations expressed through ND4J primitives. */
final class VulkanLevel2 implements Level2 {
    private static boolean transposed(char trans) {
        return trans == 'T' || trans == 't' || trans == 'C' || trans == 'c';
    }

    private static INDArray triangular(INDArray a, char uplo, char diag) {
        boolean upper = uplo == 'U' || uplo == 'u';
        INDArray out = upper
                ? Nd4j.getExecutioner().exec(new Triu(a, 0))[0]
                : Nd4j.getExecutioner().exec(new Triu(a.transpose(), 0))[0].transpose();
        if (diag == 'U' || diag == 'u') {
            INDArray identity = Nd4j.eye(out.rows()).castTo(out.dataType());
            out = out.mul(Nd4j.onesLike(out).sub(identity)).add(identity);
        }
        return out;
    }

    private static INDArray symmetric(INDArray a, char uplo) {
        INDArray triangular = triangular(a, uplo, 'N');
        INDArray diagonal = triangular.mul(Nd4j.eye(a.rows()).castTo(a.dataType()));
        return triangular.add(triangular.transpose()).sub(diagonal);
    }

    private static int packedOrder(INDArray packed) {
        long length = packed.length();
        int n = (int) ((Math.sqrt(8.0 * length + 1.0) - 1.0) * 0.5);
        if ((long) n * (n + 1) / 2 != length) {
            throw new IllegalArgumentException("Packed BLAS storage length must be triangular");
        }
        return n;
    }

    private static INDArray unpackPacked(INDArray packed, char uplo, char diag) {
        int n = packedOrder(packed);
        INDArray matrix = Nd4j.zeros(packed.dataType(), n, n);
        long index = 0;
        boolean upper = uplo == 'U' || uplo == 'u';
        for (int column = 0; column < n; column++) {
            int first = upper ? 0 : column;
            int last = upper ? column : n - 1;
            for (int row = first; row <= last; row++) {
                matrix.putScalar(row, column, packed.getDouble(index++));
            }
        }
        if (diag == 'U' || diag == 'u') {
            INDArray identity = Nd4j.eye(n).castTo(packed.dataType());
            matrix = matrix.mul(Nd4j.onesLike(matrix).sub(identity)).add(identity);
        }
        return matrix;
    }

    private static void packSymmetric(INDArray matrix, INDArray packed, char uplo) {
        int n = packedOrder(packed);
        long index = 0;
        boolean upper = uplo == 'U' || uplo == 'u';
        for (int column = 0; column < n; column++) {
            int first = upper ? 0 : column;
            int last = upper ? column : n - 1;
            for (int row = first; row <= last; row++) {
                packed.putScalar(index++, matrix.getDouble(row, column));
            }
        }
    }

    private static void assignTriangle(INDArray target, INDArray value, char uplo) {
        boolean upper = uplo == 'U' || uplo == 'u';
        INDArray ones = Nd4j.onesLike(target);
        INDArray mask = upper
                ? Nd4j.getExecutioner().exec(new Triu(ones, 0))[0]
                : Nd4j.getExecutioner().exec(new Triu(ones.transpose(), 0))[0].transpose();
        target.assign(target.mul(ones.sub(mask)).add(value.mul(mask)));
    }

    private static void gemvImpl(char trans, double alpha, INDArray a, INDArray x,
                                 double beta, INDArray y) {
        INDArray matrix = transposed(trans) ? a.transpose() : a;
        INDArray result = matrix.mmul(x.reshape(x.length(), 1)).reshape(y.shape());
        y.assign(result.mul(alpha).add(y.mul(beta)));
    }

    @Override
    public void gemv(char order, char transA, double alpha, INDArray a, INDArray x,
                     double beta, INDArray y) {
        gemvImpl(transA, alpha, a, x, beta, y);
    }

    @Override
    public void gbmv(char order, char transA, int kl, int ku, double alpha,
                     INDArray a, INDArray x, double beta, INDArray y) {
        gemvImpl(transA, alpha, a, x, beta, y);
    }

    @Override
    public void ger(char order, double alpha, INDArray x, INDArray y, INDArray a) {
        INDArray update = x.reshape(x.length(), 1).mmul(y.reshape(1, y.length())).mul(alpha);
        a.assign(a.add(update));
    }

    @Override
    public void sbmv(char order, char uplo, double alpha, INDArray a, INDArray x,
                     double beta, INDArray y) {
        gemvImpl('N', alpha, symmetric(a, uplo), x, beta, y);
    }

    @Override
    public void spmv(char order, char uplo, double alpha, INDArray ap, INDArray x,
                     double beta, INDArray y) {
        gemvImpl('N', alpha, unpackPacked(ap, uplo, 'N'), x, beta, y);
    }

    @Override
    public void spr(char order, char uplo, double alpha, INDArray x, INDArray ap) {
        INDArray matrix = unpackPacked(ap, uplo, 'N');
        INDArray update = x.reshape(x.length(), 1).mmul(x.reshape(1, x.length())).mul(alpha);
        packSymmetric(matrix.add(update), ap, uplo);
    }

    @Override
    public void spr2(char order, char uplo, double alpha, INDArray x, INDArray y, INDArray a) {
        INDArray matrix = unpackPacked(a, uplo, 'N');
        INDArray update = x.reshape(x.length(), 1).mmul(y.reshape(1, y.length()))
                .add(y.reshape(y.length(), 1).mmul(x.reshape(1, x.length()))).mul(alpha);
        packSymmetric(matrix.add(update), a, uplo);
    }

    @Override
    public void symv(char order, char uplo, double alpha, INDArray a, INDArray x,
                     double beta, INDArray y) {
        gemvImpl('N', alpha, symmetric(a, uplo), x, beta, y);
    }

    @Override
    public void syr(char order, char uplo, int n, double alpha, INDArray x, INDArray a) {
        INDArray update = x.reshape(x.length(), 1).mmul(x.reshape(1, x.length())).mul(alpha);
        assignTriangle(a, a.add(update), uplo);
    }

    @Override
    public void syr2(char order, char uplo, double alpha, INDArray x, INDArray y, INDArray a) {
        INDArray update = x.reshape(x.length(), 1).mmul(y.reshape(1, y.length()))
                .add(y.reshape(y.length(), 1).mmul(x.reshape(1, x.length()))).mul(alpha);
        assignTriangle(a, a.add(update), uplo);
    }

    @Override
    public void tbmv(char order, char uplo, char transA, char diag, INDArray a, INDArray x) {
        gemvImpl(transA, 1.0, triangular(a, uplo, diag), x, 0.0, x);
    }

    @Override
    public void tbsv(char order, char uplo, char transA, char diag, INDArray a, INDArray x) {
        trsv(order, uplo, transA, diag, a, x);
    }

    @Override
    public void tpmv(char order, char uplo, char transA, char diag, INDArray ap, INDArray x) {
        gemvImpl(transA, 1.0, unpackPacked(ap, uplo, diag), x, 0.0, x);
    }

    @Override
    public void tpsv(char order, char uplo, char transA, char diag, INDArray ap, INDArray x) {
        trsv(order, uplo, transA, diag, unpackPacked(ap, uplo, diag), x);
    }

    @Override
    public void trmv(char order, char uplo, char transA, char diag, INDArray a, INDArray x) {
        tbmv(order, uplo, transA, diag, a, x);
    }

    @Override
    public void trsv(char order, char uplo, char transA, char diag, INDArray a, INDArray x) {
        INDArray result = Nd4j.getExecutioner().exec(new TriangularSolve(
                triangular(a, uplo, diag), x, uplo == 'L' || uplo == 'l', transposed(transA)))[0];
        x.assign(result);
    }
}
