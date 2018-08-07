/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.blas;

import org.nd4j.nativeblas.Nd4jBlas;

import static org.bytedeco.javacpp.cublas.*;

/**
 * Implementation of Nd4jBlas for cuBLAS
 *
 * @author saudet
 */
public class CudaBlas extends Nd4jBlas {

    static int convertStatus(int status) {
        switch (status) {
            case 0:
                return CUBLAS_STATUS_SUCCESS;
            case 1:
                return CUBLAS_STATUS_NOT_INITIALIZED;
            case 3:
                return CUBLAS_STATUS_ALLOC_FAILED;
            case 7:
                return CUBLAS_STATUS_INVALID_VALUE;
            case 8:
                return CUBLAS_STATUS_ARCH_MISMATCH;
            case 11:
                return CUBLAS_STATUS_MAPPING_ERROR;
            case 13:
                return CUBLAS_STATUS_EXECUTION_FAILED;
            case 14:
                return CUBLAS_STATUS_INTERNAL_ERROR;
            case 15:
                return CUBLAS_STATUS_NOT_SUPPORTED;
            case 16:
                return CUBLAS_STATUS_LICENSE_ERROR;
            default:
                return CUBLAS_STATUS_SUCCESS;
        }
    }

    static int convertUplo(int fillMode) {
        switch (fillMode) {
            case 0:
                return CUBLAS_FILL_MODE_LOWER;
            case 1:
                return CUBLAS_FILL_MODE_UPPER;
            default:
                return CUBLAS_FILL_MODE_LOWER;
        }
    }

    static int convertDiag(int diag) {
        switch (diag) {
            case 0:
                return CUBLAS_DIAG_NON_UNIT;
            case 1:
                return CUBLAS_DIAG_UNIT;
            default:
                return CUBLAS_DIAG_NON_UNIT;
        }
    }

    static int convertTranspose(int op) {
        switch (op) {
            case 78:
                return CUBLAS_OP_N;
            case 84:
                return CUBLAS_OP_T;
            case 67:
                return CUBLAS_OP_C;
            default:
                return CUBLAS_OP_N;
        }
    }

    static int convertPointerMode(int pointerMode) {
        switch (pointerMode) {
            case 0:
                return CUBLAS_POINTER_MODE_HOST;
            case 1:
                return CUBLAS_POINTER_MODE_DEVICE;
            default:
                return CUBLAS_POINTER_MODE_HOST;
        }
    }

    static int convertSideMode(int sideMode) {
        switch (sideMode) {
            case 0:
                return CUBLAS_SIDE_LEFT;
            case 1:
                return CUBLAS_SIDE_RIGHT;
            default:
                return CUBLAS_SIDE_LEFT;
        }
    }

    @Override
    public void setMaxThreads(int num) {
        // no-op
    }

    @Override
    public int getMaxThreads() {
        // 0 - cuBLAS
        return 0;
    }

    /**
     * Returns the BLAS library vendor id
     *
     * 1 - CUBLAS
     *
     * @return the BLAS library vendor id
     */
    @Override
    public int getBlasVendorId() {
        return 1;
    }
}
