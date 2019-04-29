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

//
//  @author raver119@gmail.com
//

#include <helpers/BlasHelper.h>
namespace nd4j {
    BlasHelper* BlasHelper::getInstance() {
        if (_instance == 0)
            _instance = new BlasHelper();
        return _instance;
    }


    void BlasHelper::initializeFunctions(Nd4jPointer *functions) {
        nd4j_debug("Initializing BLAS\n","");

        _hasSgemv = functions[0] != nullptr;
        _hasSgemm = functions[2] != nullptr;

        _hasDgemv = functions[1] != nullptr;
        _hasDgemm = functions[3] != nullptr;

        _hasSgemmBatch = functions[4] != nullptr;
        _hasDgemmBatch = functions[5] != nullptr;

        this->cblasSgemv = (CblasSgemv)functions[0];
        this->cblasDgemv = (CblasDgemv)functions[1];
        this->cblasSgemm = (CblasSgemm)functions[2];
        this->cblasDgemm = (CblasDgemm)functions[3];
        this->cblasSgemmBatch = (CblasSgemmBatch)functions[4];
        this->cblasDgemmBatch = (CblasDgemmBatch)functions[5];
        this->lapackeSgesvd = (LapackeSgesvd)functions[6];
        this->lapackeDgesvd = (LapackeDgesvd)functions[7];
        this->lapackeSgesdd = (LapackeSgesdd)functions[8];
        this->lapackeDgesdd = (LapackeDgesdd)functions[9];
    }

    void BlasHelper::initializeDeviceFunctions(Nd4jPointer *functions) {
        nd4j_debug("Initializing device BLAS\n","");

        /*
	    this->cublasSgemv = (CublasSgemv)functions[0];
        this->cublasDgemv = (CublasDgemv)functions[1];
        this->cublasHgemm = (CublasHgemm)functions[2];
        this->cublasSgemm = (CublasSgemm)functions[3];
        this->cublasDgemm = (CublasDgemm)functions[4];
        this->cublasSgemmEx = (CublasSgemmEx)functions[5];
        this->cublasHgemmBatched = (CublasHgemmBatched)functions[6];
        this->cublasSgemmBatched = (CublasSgemmBatched)functions[7];
        this->cublasDgemmBatched = (CublasDgemmBatched)functions[8];
        this->cusolverDnSgesvdBufferSize = (CusolverDnSgesvdBufferSize)functions[9];
        this->cusolverDnDgesvdBufferSize = (CusolverDnDgesvdBufferSize)functions[10];
        this->cusolverDnSgesvd = (CusolverDnSgesvd)functions[11];
        this->cusolverDnDgesvd = (CusolverDnDgesvd)functions[12];
	    */
    }


    template <>
    bool BlasHelper::hasGEMV<float>() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return true;
#else
        return _hasSgemv;
#endif
    }

    template <>
    bool BlasHelper::hasGEMV<double>() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return true;
#else
    return _hasDgemv;
#endif
    }

    template <>
    bool BlasHelper::hasGEMV<float16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<bfloat16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<bool>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<int>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<int8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<uint8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<int16_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMV<Nd4jLong>() {
        return false;
    }

    bool BlasHelper::hasGEMV(const nd4j::DataType dtype)  {
        if(dtype == DataType::FLOAT32) {
            #if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
                return true;
            #else
                return _hasSgemv;
            #endif
        }
        if(dtype == DataType::DOUBLE) {
            #if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
                return true;
            #else
                return _hasDgemv;
            #endif
        }
        return false;        
    }

    template <>
    bool BlasHelper::hasGEMM<float>() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return true;
#else
    return _hasSgemm;
#endif
    }

    template <>
    bool BlasHelper::hasGEMM<double>() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return true;
#else
    return _hasDgemm;
#endif
    }

    template <>
    bool BlasHelper::hasGEMM<float16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<bfloat16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<int>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<uint8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<int8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<int16_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<bool>() {
        return false;
    }

    template <>
    bool BlasHelper::hasGEMM<Nd4jLong>() {
        return false;
    }

    bool BlasHelper:: hasGEMM(const nd4j::DataType dtype) {
        if(dtype == DataType::FLOAT32) {
            #if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
                return true;
            #else
                return _hasSgemm;
            #endif
        }
        if(dtype == DataType::DOUBLE) {
            #if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
                return true;
            #else
                return _hasDgemm;
            #endif
        }
        return false;
    }


    template <>
    bool BlasHelper::hasBatchedGEMM<float>() {
        return _hasSgemmBatch;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<double>() {
        return _hasDgemmBatch;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<float16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<bfloat16>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<Nd4jLong>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<int>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<int8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<uint8_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<int16_t>() {
        return false;
    }

    template <>
    bool BlasHelper::hasBatchedGEMM<bool>() {
        return false;
    }

    CblasSgemv BlasHelper::sgemv() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return (CblasSgemv)&cblas_sgemv;
#else
        return this->cblasSgemv;
#endif
    }
    CblasDgemv BlasHelper::dgemv() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return (CblasDgemv)&cblas_dgemv;
#else
        return this->cblasDgemv;
#endif
    }

    CblasSgemm BlasHelper::sgemm() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return (CblasSgemm)&cblas_sgemm;
#else
        return this->cblasSgemm;
#endif
    }

    CblasDgemm BlasHelper::dgemm() {
#if defined(__EXTERNAL_BLAS__) || defined(HAVE_MKLDNN) || defined(HAVE_OPENBLAS)
        return (CblasDgemm)&cblas_dgemm;
#else
        return this->cblasDgemm;
#endif
    }

    CblasSgemmBatch BlasHelper::sgemmBatched() {
        return this->cblasSgemmBatch;
    }

    CblasDgemmBatch BlasHelper::dgemmBatched() {
        return this->cblasDgemmBatch;
    }

    LapackeSgesvd BlasHelper::sgesvd() {
        return this->lapackeSgesvd;
    }

    LapackeDgesvd BlasHelper::dgesvd() {
        return this->lapackeDgesvd;
    }

    LapackeSgesdd BlasHelper::sgesdd() {
        return this->lapackeSgesdd;
    }

    LapackeDgesdd BlasHelper::dgesdd() {
        return this->lapackeDgesdd;
    }

    // destructor
    BlasHelper::~BlasHelper() noexcept { }

    BlasHelper* BlasHelper::_instance = 0;
}
