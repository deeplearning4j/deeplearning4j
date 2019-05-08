/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
//


#include <cublas_v2.h>
#include "../cublasHelper.h"
//#include <graph/exceptions/cuda_exception.h>
#include <stdexcept>

namespace nd4j {
    void* cublas::handle() {
        //auto _handle = new cublasHandle_t();
        //auto status = cublasCreate_v2(_handle); // initialize CUBLAS context
        //if (status != CUBLAS_STATUS_SUCCESS)
            //throw cuda_exception::build("cuBLAS handle creation failed !", status);
        //    throw std::runtime_error("cuBLAS handle creation failed !");

        return nullptr;
    }
}