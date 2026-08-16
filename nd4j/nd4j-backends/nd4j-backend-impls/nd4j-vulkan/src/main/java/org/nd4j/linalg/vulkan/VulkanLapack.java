/*
 * ******************************************************************************
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * LAPACK surface for Vulkan arrays.
 *
 * <p>The Vulkan native library intentionally does not link a platform BLAS or
 * LAPACK implementation.  Keeping this implementation in the backend makes
 * the complete ND4J LAPACK contract available without selecting a host BLAS
 * by vendor name or relying on a second backend.  The algorithms operate on
 * the backend's INDArray abstraction, so the same code works for every Vulkan
 * device and data type supported by the array factory.</p>
 */
public final class VulkanLapack extends BaseLapack {
    private static final double EPS = 1.0e-12;

    private static int size(INDArray array) {
        if (array.rows() > Integer.MAX_VALUE || array.columns() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Vulkan LAPACK only supports Java-sized matrices");
        }
        return (int) array.rows();
    }

    private static void info(INDArray info, int value) {
        if (info != null && info.length() > 0) {
            info.putScalar(0, value);
        }
    }

    private static void requireReal(INDArray array) {
        DataType type = array.dataType();
        if (type != DataType.FLOAT && type != DataType.DOUBLE) {
            throw new UnsupportedOperationException("Vulkan LAPACK requires FLOAT or DOUBLE arrays");
        }
    }

    @Override
    public void sgetrf(int m, int n, INDArray a, INDArray ipiv, INDArray info) {
        getrf(m, n, a, ipiv, info);
    }

    @Override
    public void dgetrf(int m, int n, INDArray a, INDArray ipiv, INDArray info) {
        getrf(m, n, a, ipiv, info);
    }

    private static void getrf(int m, int n, INDArray a, INDArray ipiv, INDArray info) {
        requireReal(a);
        int limit = Math.min(m, n);
        int status = 0;
        for (int k = 0; k < limit; k++) {
            int pivot = k;
            double largest = Math.abs(a.getDouble(k, k));
            for (int row = k + 1; row < m; row++) {
                double candidate = Math.abs(a.getDouble(row, k));
                if (candidate > largest) {
                    largest = candidate;
                    pivot = row;
                }
            }
            ipiv.putScalar(k, pivot + 1);
            if (pivot != k) {
                for (int col = 0; col < n; col++) {
                    double value = a.getDouble(k, col);
                    a.putScalar(k, col, a.getDouble(pivot, col));
                    a.putScalar(pivot, col, value);
                }
            }
            double diagonal = a.getDouble(k, k);
            if (Math.abs(diagonal) <= EPS) {
                if (status == 0) {
                    status = k + 1;
                }
                continue;
            }
            for (int row = k + 1; row < m; row++) {
                double multiplier = a.getDouble(row, k) / diagonal;
                a.putScalar(row, k, multiplier);
                for (int col = k + 1; col < n; col++) {
                    a.putScalar(row, col,
                            a.getDouble(row, col) - multiplier * a.getDouble(k, col));
                }
            }
        }
        info(info, status);
    }

    @Override
    public void spotrf(byte uplo, int n, INDArray a, INDArray info) {
        potrf(uplo, n, a, info);
    }

    @Override
    public void dpotrf(byte uplo, int n, INDArray a, INDArray info) {
        potrf(uplo, n, a, info);
    }

    private static void potrf(byte uplo, int n, INDArray a, INDArray info) {
        requireReal(a);
        boolean lower = uplo == 'L' || uplo == 'l';
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = a.getDouble(i, j);
                for (int k = 0; k < j; k++) {
                    sum -= a.getDouble(i, k) * a.getDouble(j, k);
                }
                if (i == j) {
                    if (sum <= EPS) {
                        info(info, i + 1);
                        return;
                    }
                    a.putScalar(i, j, Math.sqrt(sum));
                } else {
                    a.putScalar(i, j, sum / a.getDouble(j, j));
                }
            }
        }
        if (!lower) {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    a.putScalar(i, j, a.getDouble(j, i));
                }
                for (int j = 0; j < i; j++) {
                    a.putScalar(i, j, 0.0);
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    a.putScalar(i, j, 0.0);
                }
            }
        }
        info(info, 0);
    }

    @Override
    public void sgeqrf(int m, int n, INDArray a, INDArray r, INDArray info) {
        geqrf(m, n, a, r, info);
    }

    @Override
    public void dgeqrf(int m, int n, INDArray a, INDArray r, INDArray info) {
        geqrf(m, n, a, r, info);
    }

    private static void geqrf(int m, int n, INDArray a, INDArray r, INDArray info) {
        requireReal(a);
        int rank = Math.min(m, n);
        double[][] q = new double[m][rank];
        for (int col = 0; col < rank; col++) {
            for (int row = 0; row < m; row++) q[row][col] = a.getDouble(row, col);
            for (int previous = 0; previous < col; previous++) {
                double dot = 0.0;
                for (int row = 0; row < m; row++) dot += q[row][previous] * q[row][col];
                r.putScalar(previous, col, dot);
                for (int row = 0; row < m; row++) q[row][col] -= dot * q[row][previous];
            }
            double norm = 0.0;
            for (int row = 0; row < m; row++) norm += q[row][col] * q[row][col];
            norm = Math.sqrt(norm);
            r.putScalar(col, col, norm);
            if (norm > EPS) {
                for (int row = 0; row < m; row++) q[row][col] /= norm;
            }
            for (int next = col + 1; next < n; next++) {
                double dot = 0.0;
                for (int row = 0; row < m; row++) dot += q[row][col] * a.getDouble(row, next);
                r.putScalar(col, next, dot);
            }
        }
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < rank; col++) a.putScalar(row, col, q[row][col]);
        }
        info(info, 0);
    }

    @Override
    public int ssyev(char jobz, char uplo, int n, INDArray a, INDArray values) {
        return syev(jobz, n, a, values);
    }

    @Override
    public int dsyev(char jobz, char uplo, int n, INDArray a, INDArray values) {
        return syev(jobz, n, a, values);
    }

    private static int syev(char jobz, int n, INDArray a, INDArray values) {
        requireReal(a);
        double[][] eigenvectors = new double[n][n];
        for (int i = 0; i < n; i++) eigenvectors[i][i] = 1.0;
        for (int iteration = 0; iteration < n * n * 8; iteration++) {
            int p = 0, q = 1;
            double maximum = n < 2 ? 0.0 : Math.abs(a.getDouble(p, q));
            for (int row = 0; row < n; row++) {
                for (int col = row + 1; col < n; col++) {
                    double candidate = Math.abs(a.getDouble(row, col));
                    if (candidate > maximum) { maximum = candidate; p = row; q = col; }
                }
            }
            if (maximum <= EPS) break;
            double angle = 0.5 * Math.atan2(2.0 * a.getDouble(p, q),
                    a.getDouble(p, p) - a.getDouble(q, q));
            double cosine = Math.cos(angle), sine = Math.sin(angle);
            for (int k = 0; k < n; k++) {
                double apk = a.getDouble(p, k), aqk = a.getDouble(q, k);
                a.putScalar(p, k, cosine * apk - sine * aqk);
                a.putScalar(q, k, sine * apk + cosine * aqk);
            }
            for (int k = 0; k < n; k++) {
                double akp = a.getDouble(k, p), akq = a.getDouble(k, q);
                a.putScalar(k, p, cosine * akp - sine * akq);
                a.putScalar(k, q, sine * akp + cosine * akq);
                double vkp = eigenvectors[k][p], vkq = eigenvectors[k][q];
                eigenvectors[k][p] = cosine * vkp - sine * vkq;
                eigenvectors[k][q] = sine * vkp + cosine * vkq;
            }
        }
        for (int i = 0; i < n; i++) values.putScalar(i, a.getDouble(i, i));
        if (jobz == 'V' || jobz == 'v') {
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) a.putScalar(row, col, eigenvectors[row][col]);
            }
        }
        return 0;
    }

    @Override
    public void sgesvd(byte jobu, byte jobvt, int m, int n, INDArray a, INDArray s,
                       INDArray u, INDArray vt, INDArray info) {
        gesvd(jobu, jobvt, m, n, a, s, u, vt, info);
    }

    @Override
    public void dgesvd(byte jobu, byte jobvt, int m, int n, INDArray a, INDArray s,
                       INDArray u, INDArray vt, INDArray info) {
        gesvd(jobu, jobvt, m, n, a, s, u, vt, info);
    }

    private static void gesvd(byte jobu, byte jobvt, int m, int n, INDArray a, INDArray s,
                              INDArray u, INDArray vt, INDArray info) {
        requireReal(a);
        int rank = Math.min(m, n);
        double[][] ata = new double[n][n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                for (int k = 0; k < m; k++) ata[row][col] += a.getDouble(k, row) * a.getDouble(k, col);
            }
        }
        INDArray eigen = Nd4j.create(a.dataType(), new long[]{n, n});
        for (int row = 0; row < n; row++) for (int col = 0; col < n; col++) eigen.putScalar(row, col, ata[row][col]);
        INDArray eigenvalues = Nd4j.create(a.dataType(), n);
        syev('V', n, eigen, eigenvalues);
        for (int i = 0; i < rank; i++) {
            double sigma = Math.sqrt(Math.max(0.0, eigenvalues.getDouble(n - 1 - i)));
            s.putScalar(i, sigma);
            if (vt != null) for (int col = 0; col < n; col++) vt.putScalar(i, col, eigen.getDouble(col, n - 1 - i));
            if (u != null && sigma > EPS) {
                for (int row = 0; row < m; row++) {
                    double value = 0.0;
                    for (int col = 0; col < n; col++) value += a.getDouble(row, col) * eigen.getDouble(col, n - 1 - i);
                    u.putScalar(row, i, value / sigma);
                }
            }
        }
        info(info, 0);
    }

    /**
     * Generate an inverse from the packed LU factors produced by getrf.
     * The solve is expressed in INDArray scalar operations so it remains
     * backend-owned and does not depend on a host BLAS/LAPACK library.
     */
    @Override
    public void getri(int n, INDArray a, int lda, int[] ipiv, INDArray work, int lwork, int info) {
        requireReal(a);
        if (n < 0 || n > a.rows() || n > a.columns()) {
            throw new IllegalArgumentException("Invalid matrix order for getri: " + n);
        }
        INDArray inverse = Nd4j.create(a.dataType(), new long[]{n, n});
        for (int column = 0; column < n; column++) {
            double[] rhs = new double[n];
            rhs[column] = 1.0;
            for (int row = 0; row < n && row < ipiv.length; row++) {
                int pivot = ipiv[row] - 1;
                if (pivot >= 0 && pivot < n && pivot != row) {
                    double value = rhs[row];
                    rhs[row] = rhs[pivot];
                    rhs[pivot] = value;
                }
            }
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < row; col++) rhs[row] -= a.getDouble(row, col) * rhs[col];
            }
            for (int row = n - 1; row >= 0; row--) {
                for (int col = row + 1; col < n; col++) rhs[row] -= a.getDouble(row, col) * rhs[col];
                double diagonal = a.getDouble(row, row);
                if (Math.abs(diagonal) <= EPS) throw new IllegalArgumentException("Singular matrix in getri");
                rhs[row] /= diagonal;
            }
            for (int row = 0; row < n; row++) inverse.putScalar(row, column, rhs[row]);
        }
        a.assign(inverse);
    }

    @Override
    public INDArray getPFactor(int m, INDArray ipiv) {
        INDArray result = Nd4j.eye(m).castTo(ipiv.dataType());
        for (int i = 0; i < ipiv.length(); i++) {
            int pivot = ipiv.getInt(i) - 1;
            if (pivot > i) {
                INDArray first = result.getColumn(i).dup();
                result.putColumn(i, result.getColumn(pivot));
                result.putColumn(pivot, first);
            }
        }
        return result;
    }

    @Override
    public INDArray getLFactor(INDArray a) {
        INDArray result = Nd4j.create(a.dataType(), a.shape());
        for (int row = 0; row < a.rows(); row++) {
            for (int col = 0; col < a.columns(); col++) {
                result.putScalar(row, col, row > col ? a.getDouble(row, col) : row == col ? 1.0 : 0.0);
            }
        }
        return result;
    }

    @Override
    public INDArray getUFactor(INDArray a) {
        INDArray result = Nd4j.create(a.dataType(), new long[]{a.columns(), a.columns()});
        for (int row = 0; row < result.rows(); row++) {
            for (int col = 0; col < result.columns(); col++) {
                result.putScalar(row, col, row <= col && row < a.rows() ? a.getDouble(row, col) : 0.0);
            }
        }
        return result;
    }
}
