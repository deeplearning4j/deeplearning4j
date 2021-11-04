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

#include <cstdarg>
#include <vector>

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
#define sd_printv(FORMAT, VECTOR) sd::Logger::printv(FORMAT, VECTOR);

#else

#define sd_debug(FORMAT, A, ...)
#define sd_logger(FORMAT, A, ...)
#define sd_verbose(FORMAT, ...)
#define sd_printf(FORMAT, ...) sd::Logger::info(FORMAT, __VA_ARGS__);
#define sd_printv(FORMAT, VECTOR)

#endif

namespace sd {
class SD_LIB_EXPORT Logger {
 public:
  static SD_HOST void info(const char *format, ...);

  static SD_HOST void printv(const char *format, const std::vector<int> &vec);
  static SD_HOST void printv(const char *format, const std::vector<sd::LongType> &vec);

  static SD_HOST_DEVICE Status logStatusMsg(Status code, const char *msg);

  static SD_HOST_DEVICE Status logKernelFailureMsg(const char *msg = nullptr);
};

}  // namespace sd

#endif  // LIBND4J_LOGGER_H
