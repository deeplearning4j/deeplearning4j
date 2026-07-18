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
#include <array/ExtraArguments_device.h>
#include <types/types.h>

#include <stdexcept>

SD_BACKEND_ABI_NAMESPACE_BEGIN
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
    extra_args_detail::extraArgsFreeDevice(p);
  }
}

template <typename T>
void ExtraArguments::convertAndCopy(Pointer pointer, LongType offset) {
  auto length = this->length();
  if (offset < 0 || static_cast<size_t>(offset) > length) {
    THROW_EXCEPTION("ExtraArguments::convertAndCopy: offset is out of range");
  }
  auto outLength = length - static_cast<size_t>(offset);

  // Fill a local host buffer, then hand the copy to the selected backend.
  // Device backends preserve their active execution-stream ordering.
  auto hostBuf = new T[outLength];

  if (!_fpArgs.empty()) {
    for (size_t e = offset; e < _fpArgs.size(); e++) {
      hostBuf[e - static_cast<size_t>(offset)] = static_cast<T>(_fpArgs[e]);
    }
  } else if (!_intArgs.empty()) {
    for (size_t e = offset; e < _intArgs.size(); e++) {
      hostBuf[e - static_cast<size_t>(offset)] = static_cast<T>(_intArgs[e]);
    }
  }

  auto bytes = outLength * DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>());
  extra_args_detail::extraArgsCopyH2DDispatch(pointer, hostBuf, bytes);
  delete[] hostBuf;
}
BUILD_SINGLE_TEMPLATE(void ExtraArguments::convertAndCopy,
                      (sd::Pointer pointer, sd::LongType offset), SD_COMMON_TYPES);

void *ExtraArguments::allocate(size_t length, size_t elementSize) {
  auto ptr = extra_args_detail::extraArgsAllocDevice(length * elementSize);
  if (!ptr) THROW_EXCEPTION("Can't allocate memory");
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
SD_BACKEND_ABI_NAMESPACE_END
