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
//

#include <array/ExtraArguments.h>
#include <array/DataType.h>
#include <array/DataTypeUtils.h>
#include <stdexcept>

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace nd4j {
    ExtraArguments::ExtraArguments(std::initializer_list<double> arguments) {
        _fpArgs = arguments;
    }

    ExtraArguments::ExtraArguments(std::initializer_list<Nd4jLong> arguments) {
        _intArgs = arguments;
    }

    ExtraArguments::ExtraArguments() {

    }

    ExtraArguments::~ExtraArguments() {
        for (auto p:_pointers) {
#ifdef __CUDABLAS__
            cudaFree(p);
#else // CPU branch
            free(p);
#endif
        }
    }

    void* ExtraArguments::allocate(size_t length, size_t elementSize) {
#ifdef __CUDABLAS__
        Nd4jPointer ptr;
	    auto res = cudaMalloc(reinterpret_cast<void **>(&ptr), length * elementSize);
	    if (res != 0)
		    throw std::runtime_error("Can't allocate CUDA memory");
#else // CPU branch
        auto ptr = (Nd4jPointer) malloc(length * elementSize);
        if (!ptr)
            throw std::runtime_error("Can't allocate memory");
#endif

        return ptr;
    }

    size_t ExtraArguments::length() {
        if (!_fpArgs.empty())
            return _fpArgs.size();
        else if (!_intArgs.empty())
            return _intArgs.size();
        else
            return 0;
    }

    template <typename T>
    void* ExtraArguments::argumentsAsT() {
        return argumentAsT(DataTypeUtils::fromT<T>());
    }

    void* ExtraArguments::argumentAsT(nd4j::DataType dataType) {
        if (_fpArgs.empty() && _intArgs.empty())
            return nullptr;

        // we allocate pointer
        auto ptr = allocate(length(), DataTypeUtils::sizeOf(dataType));

        // fill it with data

        // store it internally for future release
        _pointers.emplace_back(ptr);

        return ptr;
    }
}
