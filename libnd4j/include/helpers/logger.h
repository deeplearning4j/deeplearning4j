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
// Created by raver119 on 09.01.17.
//

#ifndef LIBND4J_LOGGER_H
#define LIBND4J_LOGGER_H
#include <stdio.h>
#include <stdlib.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <cstdarg>
#include <vector>
#include <limits>

#include <iostream>
#include <string>
#include <typeinfo>
#include <iomanip>
#include <sstream>
#include <bitset>
#include <cassert>
#include <type_traits>

#ifndef __CUDA_ARCH__

#define sd_debug(FORMAT, ...)                                                                 \
  if (sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()) \
    sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_logger(FORMAT, ...)                                                                \
  if (sd::Environment::getInstance().isDebug() && sd::Environment::getInstance().isVerbose()) \
    sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_verbose(FORMAT, ...) \
  if (sd::Environment::getInstance().isVerbose()) sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_printf(FORMAT, ...) sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_print(FORMAT) sd::Logger::infoEmpty(FORMAT);
#define sd_printv(FORMAT, VECTOR) sd::Logger::printv(FORMAT, VECTOR);

#else

#define sd_debug(FORMAT, A, ...)
#define sd_logger(FORMAT, A, ...)
#define sd_verbose(FORMAT, ...)
#define sd_printf(FORMAT, ...) sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_print(FORMAT) sd::Logger::infoEmpty(FORMAT);
#define sd_printv(FORMAT, VECTOR)

#endif


/**
 * Templated function to print the contents of a host buffer.
 *
 * @param T         The data type of the buffer elements.
 * @param buffer     Pointer to the buffer containing the data.
 * @param dataType   The data type of the elements in the buffer.
 * @param length     The total number of elements in the buffer.
 * @param offset     The starting index from which to begin printing.
 */
template <typename T>
SD_INLINE void _printHostBuffer(void* bufferVoid, sd::DataType dataType, sd::LongType length, sd::LongType offset) {
  T *buffer = reinterpret_cast<T *>(bufferVoid);
  // Validate offset
  if (offset < 0 || offset >= length) {
    printf("Invalid offset: %lld. Must be between 0 and %lld.\n", offset, length - 1);
    fflush(stdout);
    return;
  }

  // Determine the limit based on the provided length and offset
  sd::LongType limit = (length == -1) ? length : length;

  printf("[");

  // Iterate from offset to limit
  for (sd::LongType e = offset; e < limit; e++) {
    if (e > offset) {
      printf(", ");
    }

    if (dataType == sd::DataType::DOUBLE) {
      printf("%.15f", static_cast<double>(buffer[e]));
    } else if (dataType == sd::DataType::FLOAT32) {
      printf("%.15f", static_cast<float>(buffer[e]));
    } else if (dataType == sd::DataType::INT64 || dataType == sd::DataType::UINT64) {
      printf("%lld", static_cast<long long>(buffer[e]));
    } else if (dataType == sd::DataType::INT32 || dataType == sd::DataType::UINT32) {
      printf("%d", static_cast<int>(buffer[e]));
    } else if (dataType == sd::DataType::BOOL) {
      printf(static_cast<bool>(buffer[e]) ? "true" : "false");
    }

    printf("]\n");
    fflush(stdout);
  }
}

/**
 * Prints the contents of a host buffer based on the specified data type.
 *
 * @param buffer     Pointer to the buffer containing the data.
 * @param dataType   The data type of the elements in the buffer.
 * @param length     The total number of elements in the buffer.
 * @param offset     The starting index from which to begin printing.
 */
SD_INLINE void printBuffer(void* buffer, sd::DataType dataType, sd::LongType length, sd::LongType offset) {
  // Invoke the BUILD_SINGLE_SELECTOR macro to instantiate and call _printHostBuffer with the appropriate type
  //T* buffer, sd::DataType dataType, sd::LongType length, sd::LongType offset
  BUILD_SINGLE_SELECTOR(dataType, _printHostBuffer,(buffer,dataType,length,offset),SD_COMMON_TYPES_ALL);
}

namespace sd {
class SD_LIB_EXPORT Logger {
 public:
  static SD_HOST void info(const char *fdataTypeormat, ...);
  static SD_HOST void infoEmpty(const char *format);

  static SD_HOST void printv(const char *format, const std::vector<int> &vec);
  static SD_HOST void printv(const char *format, const std::vector<sd::LongType> &vec);

  static SD_HOST_DEVICE Status logStatusMsg(Status code, const char *msg);

  static SD_HOST_DEVICE Status logKernelFailureMsg(const char *msg = nullptr);
};

}  // namespace sd

#endif  // LIBND4J_LOGGER_H
