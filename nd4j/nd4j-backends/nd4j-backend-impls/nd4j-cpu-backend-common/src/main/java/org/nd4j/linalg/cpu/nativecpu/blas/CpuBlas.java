/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.blas;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.nativeblas.Nd4jBlas;

import static org.nd4j.linalg.cpu.nativecpu.blas.BLASDelegator.*;


/**
 * Implementation of Nd4jBlas with OpenBLAS/MKL
 *
 * @author saudet
 */
@Slf4j
public class CpuBlas extends Nd4jBlas {


    public static final int OPENBLAS_OS_WINNT = 1;
    public static final int OPENBLAS_ARCH_X86_64 = 1;
    public static final int OPENBLAS_C_GCC = 1;
    public static final int OPENBLAS___64BIT__ = 1;
    public static final int OPENBLAS_HAVE_C11 = 1;
    public static final int OPENBLAS_NEEDBUNDERSCORE = 1;
    public static final int OPENBLAS_L1_DATA_SIZE = 32768;
    public static final int OPENBLAS_L1_DATA_LINESIZE = 64;
    public static final int OPENBLAS_L2_SIZE = 262144;
    public static final int OPENBLAS_L2_LINESIZE = 64;
    public static final int OPENBLAS_DTB_DEFAULT_ENTRIES = 64;
    public static final int OPENBLAS_DTB_SIZE = 4096;
    public static final String OPENBLAS_CHAR_CORENAME = "NEHALEM";
    public static final int OPENBLAS_SLOCAL_BUFFER_SIZE = 65536;
    public static final int OPENBLAS_DLOCAL_BUFFER_SIZE = 32768;
    public static final int OPENBLAS_CLOCAL_BUFFER_SIZE = 65536;
    public static final int OPENBLAS_ZLOCAL_BUFFER_SIZE = 32768;
    public static final int OPENBLAS_GEMM_MULTITHREAD_THRESHOLD = 4;
    public static final String OPENBLAS_VERSION = " OpenBLAS 0.3.19 ";
    public static final int OPENBLAS_SEQUENTIAL = 0;
    public static final int OPENBLAS_THREAD = 1;
    public static final int OPENBLAS_OPENMP = 2;
    public static final int CblasRowMajor = 101;
    public static final int CblasColMajor = 102;
    public static final int CblasNoTrans = 111;
    public static final int CblasTrans = 112;
    public static final int CblasConjTrans = 113;
    public static final int CblasConjNoTrans = 114;
    public static final int CblasUpper = 121;
    public static final int CblasLower = 122;
    public static final int CblasNonUnit = 131;
    public static final int CblasUnit = 132;
    public static final int CblasLeft = 141;
    public static final int CblasRight = 142;


    /**
     * Converts a character
     * to its proper enum
     * for row (c) or column (f) ordering
     * default is row major
     */
    static int convertOrder(int from) {
        switch (from) {
            case 'c':
            case 'C':
                return CblasRowMajor;
            case 'f':
            case 'F':
                return CblasColMajor;
            default:
                return CblasColMajor;
        }
    }

    /**
     * Converts a character to its proper enum
     * t -> transpose
     * n -> no transpose
     * c -> conj
     */
    static int convertTranspose(int from) {
        switch (from) {
            case 't':
            case 'T':
                return CblasTrans;
            case 'n':
            case 'N':
                return CblasNoTrans;
            case 'c':
            case 'C':
                return CblasConjTrans;
            default:
                return CblasNoTrans;
        }
    }

    /**
     * Upper or lower
     * U/u -> upper
     * L/l -> lower
     *
     * Default is upper
     */
    static int convertUplo(int from) {
        switch (from) {
            case 'u':
            case 'U':
                return CblasUpper;
            case 'l':
            case 'L':
                return CblasLower;
            default:
                return CblasUpper;
        }
    }


    /**
     * For diagonals:
     * u/U -> unit
     * n/N -> non unit
     *
     * Default: unit
     */
    static int convertDiag(int from) {
        switch (from) {
            case 'u':
            case 'U':
                return CblasUnit;
            case 'n':
            case 'N':
                return CblasNonUnit;
            default:
                return CblasUnit;
        }
    }

    /**
     * Side of a matrix, left or right
     * l /L -> left
     * r/R -> right
     * default: left
     */
    static int convertSide(int from) {
        switch (from) {
            case 'l':
            case 'L':
                return CblasLeft;
            case 'r':
            case 'R':
                return CblasRight;
            default:
                return CblasLeft;
        }
    }

    @Override
    public void setMaxThreads(int num) {
    }

    @Override
    public int getMaxThreads() {
        return 0;
    }

    @Override
    public int getBlasVendorId() {
        return 0;
    }
}
