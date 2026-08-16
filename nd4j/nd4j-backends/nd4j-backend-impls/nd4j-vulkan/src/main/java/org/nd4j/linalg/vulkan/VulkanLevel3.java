/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.TriangularSolve;
import org.nd4j.linalg.api.ops.custom.Triu;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Level-3 BLAS compatibility surface for Vulkan.
 *
 * <p>Vulkan does not assume that a host BLAS (OpenBLAS, Accelerate, or rocBLAS)
 * is installed. Each operation is lowered to a registered Vulkan custom op, so
 * the same descriptor/emitter validation and device ownership rules apply to
 * BLAS calls and ordinary ND4J graph calls.</p>
 */
final class VulkanLevel3 implements Level3 {
    private static MMulTranspose transpose(boolean a, boolean b) {
        return MMulTranspose.builder().transposeA(a).transposeB(b).build();
    }

    private static INDArray mmul(INDArray a, INDArray b, INDArray c,
                                 double alpha, double beta,
                                 boolean transposeA, boolean transposeB) {
        INDArray[] outputs = Nd4j.getExecutioner().exec(new Mmul(
                a, b, c, alpha, beta, transpose(transposeA, transposeB)));
        return outputs[0];
    }

    private static INDArray scaledUpdate(INDArray c, INDArray value,
                                         double alpha, double beta) {
        c.assign(value.mul(alpha).add(c.mul(beta)));
        return c;
    }

    private static INDArray symmetric(INDArray a, char uplo) {
        boolean upper = uplo == 'U' || uplo == 'u';
        INDArray triangular = upper
                ? Nd4j.getExecutioner().exec(new Triu(a, 0))[0]
                : Nd4j.getExecutioner().exec(new Triu(a.transpose(), 0))[0].transpose();
        INDArray diagonal = triangular.mul(Nd4j.eye(a.rows()).castTo(a.dataType()));
        return triangular.add(triangular.transpose()).sub(diagonal);
    }

    @Override
    public void gemm(char order, char transA, char transB, double alpha,
                     INDArray a, INDArray b, double beta, INDArray c) {
        mmul(a, b, c, alpha, beta,
                transA == 'T' || transA == 't' || transA == 'C' || transA == 'c',
                transB == 'T' || transB == 't' || transB == 'C' || transB == 'c');
    }

    @Override
    public void gemm(INDArray a, INDArray b, INDArray c,
                     boolean transposeA, boolean transposeB,
                     double alpha, double beta) {
        mmul(a, b, c, alpha, beta, transposeA, transposeB);
    }

    @Override
    public void symm(char order, char side, char uplo, double alpha,
                     INDArray a, INDArray b, double beta, INDArray c) {
        INDArray symmetric = symmetric(a, uplo);
        INDArray product = side == 'R' || side == 'r'
                ? mmul(b, symmetric, null, 1.0, 0.0, false, false)
                : mmul(symmetric, b, null, 1.0, 0.0, false, false);
        scaledUpdate(c, product, alpha, beta);
    }

    @Override
    public void syrk(char order, char uplo, char trans, double alpha,
                     INDArray a, double beta, INDArray c) {
        boolean transpose = trans == 'T' || trans == 't' || trans == 'C' || trans == 'c';
        INDArray product = mmul(a, a, null, 1.0, 0.0, transpose, !transpose);
        scaledUpdate(c, product, alpha, beta);
    }

    @Override
    public void syr2k(char order, char uplo, char trans, double alpha,
                      INDArray a, INDArray b, double beta, INDArray c) {
        boolean transpose = trans == 'T' || trans == 't' || trans == 'C' || trans == 'c';
        INDArray first = transpose
                ? mmul(a, b, null, 1.0, 0.0, true, false)
                : mmul(a, b, null, 1.0, 0.0, false, true);
        INDArray second = transpose
                ? mmul(b, a, null, 1.0, 0.0, true, false)
                : mmul(b, a, null, 1.0, 0.0, false, true);
        scaledUpdate(c, first.add(second), alpha, beta);
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

    @Override
    public void trmm(char order, char side, char uplo, char transA,
                     char diag, double alpha, INDArray a, INDArray b, INDArray c) {
        // Materialize the requested triangular view on device, then dispatch GEMM.
        INDArray triangular = triangular(a, uplo, diag);
        boolean transpose = transA == 'T' || transA == 't' || transA == 'C' || transA == 'c';
        INDArray product = side == 'R' || side == 'r'
                ? mmul(b, triangular, null, alpha, 0.0, false, transpose)
                : mmul(triangular, b, null, alpha, 0.0, transpose, false);
        c.assign(product);
    }

    @Override
    public void trsm(char order, char side, char uplo, char transA,
                     char diag, double alpha, INDArray a, INDArray b) {
        boolean lower = uplo == 'L' || uplo == 'l';
        boolean adjoint = transA == 'T' || transA == 't' || transA == 'C' || transA == 'c';
        INDArray triangular = triangular(a, uplo, diag);
        INDArray result;
        if (side == 'R' || side == 'r') {
            // X op(A) = alpha B is equivalent to op(A)^T X^T = alpha B^T.
            result = Nd4j.getExecutioner().exec(new TriangularSolve(
                    triangular.transpose(), b.transpose().mul(alpha), !lower, adjoint))[0]
                    .transpose();
        } else {
            result = Nd4j.getExecutioner().exec(new TriangularSolve(
                    triangular, b.mul(alpha), lower, adjoint))[0];
        }
        b.assign(result);
    }
}
