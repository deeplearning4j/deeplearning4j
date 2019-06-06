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
#include <types/types.h>

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

    ExtraArguments::ExtraArguments(const std::vector<double> &arguments) {
        _fpArgs = arguments;
    }

    ExtraArguments::ExtraArguments(const std::vector<Nd4jLong> &arguments) {
        _intArgs = arguments;
    }

    ExtraArguments::ExtraArguments(const std::vector<int> &arguments) {
        for (const auto &v:arguments)
            _intArgs.emplace_back(static_cast<Nd4jLong>(v));
    }

    ExtraArguments::ExtraArguments() {
        // no-op
    }

    ExtraArguments::~ExtraArguments() {
        for (auto p:_pointers) {
#ifdef __CUDABLAS__
            cudaFree(p);
#else // CPU branch
            delete[] reinterpret_cast<int8_t *>(p);
#endif
        }
    }

    template <typename T>
    void ExtraArguments::convertAndCopy(Nd4jPointer pointer, Nd4jLong offset) {
        auto length = this->length();
        auto target = reinterpret_cast<T*>(pointer);
#ifdef __CUDABLAS__
        target = new T[length];
#endif

        if (!_fpArgs.empty()) {
            for (int e = offset; e < _fpArgs.size(); e++) {
                target[e] = static_cast<T>(_fpArgs[e]);
            }
        } else if (_intArgs.empty()) {
            for (int e = offset; e < _intArgs.size(); e++) {
                target[e] = static_cast<T>(_intArgs[e]);
            }
        }

#ifdef __CUDABLAS__
        // TODO: maybe make it asynchronous eventually?
        cudaMemcpy(pointer, target, length * DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>()), cudaMemcpyHostToDevice);
        delete[] target;
#endif
    }
    BUILD_SINGLE_TEMPLATE(template void ExtraArguments::convertAndCopy, (Nd4jPointer pointer, Nd4jLong offset), LIBND4J_TYPES);

    void* ExtraArguments::allocate(size_t length, size_t elementSize) {
#ifdef __CUDABLAS__
        Nd4jPointer ptr;
	    auto res = cudaMalloc(reinterpret_cast<void **>(&ptr), length * elementSize);
	    if (res != 0)
		    throw std::runtime_error("Can't allocate CUDA memory");
#else // CPU branch
        auto ptr = new int8_t[length * elementSize];
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
    void* ExtraArguments::argumentsAsT(Nd4jLong offset) {
        return argumentsAsT(DataTypeUtils::fromT<T>(), offset);
    }
    BUILD_SINGLE_TEMPLATE(template void *ExtraArguments::argumentsAsT, (Nd4jLong offset), LIBND4J_TYPES);


    void* ExtraArguments::argumentsAsT(nd4j::DataType dataType, Nd4jLong offset) {
        if (_fpArgs.empty() && _intArgs.empty())
            return nullptr;

        // we allocate pointer
        auto ptr = allocate(length() - offset, DataTypeUtils::sizeOf(dataType));

        // fill it with data
        BUILD_SINGLE_SELECTOR(dataType, convertAndCopy, (ptr, offset), LIBND4J_TYPES);

        // store it internally for future release
        _pointers.emplace_back(ptr);

        return ptr;
    }
}
