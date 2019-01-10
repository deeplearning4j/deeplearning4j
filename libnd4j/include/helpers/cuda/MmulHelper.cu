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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <exceptions/cuda_exception.h>
#include <cublas_v2.h>
#include "../MmulHelper.h"


namespace nd4j { 

// MXK x KxN = MxN
template<typename X, typename Y, typename Z>
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta) {

	if(A->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of A array is not equal 2 !");
	if(B->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of B array is not equal 2 !");
	if(C != nullptr && C->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of C array is not equal 2 !");

	const auto M = A->sizeAt(0);
	const auto K = A->sizeAt(1);
	const auto N = B->sizeAt(1);

	if(B->sizeAt(0) != K)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: B array has wrong number of rows !");
	if(C != nullptr && C->sizeAt(0) != M)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of rows !");
	if(C != nullptr && C->sizeAt(1) != N)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of columns !");

	if(C == nullptr)
		C = new NDArray('f', {M,N}, DataTypeUtils::fromT<Z>(), A->getContext());

	if(!A->isActualOnDeviceSide())
		A->syncToDevice();
	if(!B->isActualOnDeviceSide())
		B->syncToDevice();
	if(!C->isActualOnDeviceSide())
		C->syncToDevice();

	NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));

	if(A->ews() != 1 || A->ordering() == 'c')
		pA = pA->dup('f');
	if(B->ews() != 1 || B->ordering() == 'c')
		pB = pB->dup('f');
	if(C->ews() != 1 || C->ordering() == 'c')
		pC = pC->dup('f');

	auto xType = A->dataType();
    auto yType = B->dataType();
    auto zType = C->dataType();
     
    cublasStatus_t status;
    cublasHandle_t handle;

    status = cublasCreate(&handle); // initialize CUBLAS context
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    status = cublasSetStream_v2(handle, *A->getContext()->getCudaStream());
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

   	status = cublasDgemm(handle, 
   						CUBLAS_OP_N, CUBLAS_OP_N,
   						M, N, K,
   						&alpha, 
   						(double*)pA->getSpecialBuffer(), M,
   						(double*)pB->getSpecialBuffer(), K,
   						&beta,
   						(double*)pC->getSpecialBuffer(), M);

   	if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    cublasDestroy(handle);

    if(pC != C) {
    	C->assign(pC);
    	delete pC;
    }
    if(pA != A)
    	delete pA;
    if(pB != B)
    	delete pB;

	return C;
}


BUILD_TRIPLE_TEMPLATE(template nd4j::NDArray* MmulHelper::mmulMxM, (const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}
