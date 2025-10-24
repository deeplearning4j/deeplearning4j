/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/DataType.h>
#include <array/DataTypeUtils.h>
#include <array/ExtraArguments.h>
#include <types/types.h>

#include <stdexcept>

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace sd {
ExtraArguments::ExtraArguments(std::initializer_list<double> arguments) { _fpArgs = arguments; }

ExtraArguments::ExtraArguments(std::initializer_list<LongType> arguments) { _intArgs = arguments; }

ExtraArguments::ExtraArguments(const std::vector<double> &arguments) { _fpArgs = arguments; }

ExtraArguments::ExtraArguments(const std::vector<LongType> &arguments) { _intArgs = arguments; }

ExtraArguments::ExtraArguments(const std::vector<int> &arguments) {
  for (const auto &v : arguments) _intArgs.emplace_back(static_cast<LongType>(v));
}

ExtraArguments::ExtraArguments() {
  // no-op
}

ExtraArguments::~ExtraArguments() {
  for (auto p : _pointers) {
#ifdef SD_CUDA
    cudaFree(p);
#else  // CPU branch
    delete reinterpret_cast<int8_t *>(p);
#endif
  }
}

template <typename T>
void ExtraArguments::convertAndCopy(Pointer pointer, LongType offset) {
  auto length = this->length();
  auto target = reinterpret_cast<T *>(pointer);
#ifdef SD_CUDA
  target = new T[length];
#endif

  if (!_fpArgs.empty()) {
    for (size_t e = offset; e < _fpArgs.size(); e++) {
      target[e] = static_cast<T>(_fpArgs[e]);
    }
  } else if (_intArgs.empty()) {
    for (size_t e = offset; e < _intArgs.size(); e++) {
      target[e] = static_cast<T>(_intArgs[e]);
    }
  }

#ifdef SD_CUDA
  cudaMemcpy(pointer, target, length * DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>()), cudaMemcpyHostToDevice);
  delete[] target;
#endif
}
BUILD_SINGLE_TEMPLATE(void ExtraArguments::convertAndCopy,
                      (sd::Pointer pointer, sd::LongType offset), SD_COMMON_TYPES);

void *ExtraArguments::allocate(size_t length, size_t elementSize) {
#ifdef SD_CUDA
  Pointer ptr;
  auto res = cudaMalloc(reinterpret_cast<void **>(&ptr), length * elementSize);
  if (res != 0) THROW_EXCEPTION("Can't allocate CUDA memory");
#else  // CPU branch
  auto ptr = new int8_t[length * elementSize];
  if (!ptr) THROW_EXCEPTION("Can't allocate memory");
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
void *ExtraArguments::argumentsAsT(LongType offset) {
  return argumentsAsT(DataTypeUtils::fromT<T>(), offset);
}
BUILD_SINGLE_TEMPLATE(void *ExtraArguments::argumentsAsT, (sd::LongType offset),
                      SD_COMMON_TYPES);

void *ExtraArguments::argumentsAsT(DataType dataType, LongType offset) {
  if (_fpArgs.empty() && _intArgs.empty()) return nullptr;

  // we allocate pointer
  auto ptr = allocate(length() - offset, DataTypeUtils::sizeOf(dataType));

  // fill it with data
  BUILD_SINGLE_SELECTOR(dataType, convertAndCopy, (ptr, offset), SD_COMMON_TYPES);

  // store it internally for future release
  _pointers.emplace_back(ptr);

  return ptr;
}
}  // namespace sd
