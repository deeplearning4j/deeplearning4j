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
// Created by raver119 on 09.01.17.
//

#ifndef LIBND4J_LOGGER_H
#define LIBND4J_LOGGER_H

#include <vector>
#include <cstdarg>
#include <Environment.h>
#include <stdlib.h>
#include <stdio.h>
#include <dll.h>

#ifndef __CUDA_ARCH__

#define nd4j_debug(FORMAT, ...) if (nd4j::Environment::getInstance()->isDebug() && nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_logger(FORMAT, ...) if (nd4j::Environment::getInstance()->isDebug() && nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_verbose(FORMAT, ...) if (nd4j::Environment::getInstance()->isVerbose()) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_printf(FORMAT, ...) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_printv(FORMAT, VECTOR)     nd4j::Logger::printv(FORMAT, VECTOR);

#else

#define nd4j_debug(FORMAT, A, ...)
#define nd4j_logger(FORMAT, A, ...)
#define nd4j_verbose(FORMAT, ...)
#define nd4j_printf(FORMAT, ...) nd4j::Logger::info(FORMAT, __VA_ARGS__);
#define nd4j_printv(FORMAT, VECTOR)

#endif

namespace nd4j {
    class ND4J_EXPORT Logger {

    public:

        static void _CUDA_H info(const char *format, ...);

        static void _CUDA_H printv(const char *format, std::vector<int>& vec);
        static void _CUDA_H printv(const char *format, std::vector<Nd4jLong>& vec);
    };

}


#endif //LIBND4J_LOGGER_H
