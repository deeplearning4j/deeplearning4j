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
#ifndef SAMEDIFF_THREADS_H
#define SAMEDIFF_THREADS_H

#include <functional>
#include <openmp_pragmas.h>
#include <Environment.h>

namespace samediff {
    class Threads {
    public:
        /**
         * This function executes 1 dimensional loop for a given number of threads
         * @param function
         * @param numThreads
         * @param start
         * @param stop
         * @param increment
         * @return
         */
        static int parallel_for(FUNC_1D function, uint64_t start, uint64_t stop, uint64_t increment = 1, uint32_t numThreads = nd4j::Environment::getInstance()->maxThreads());

        /**
         *
         * @param function
         * @param numThreads
         * @param start_x
         * @param stop_x
         * @param inc_x
         * @param start_y
         * @param stop_y
         * @param inc_y
         * @return
         */
        static int parallel_for(FUNC_2D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t numThreads = nd4j::Environment::getInstance()->maxThreads());

        /**
         *
         * @param function
         * @param numThreads
         * @param start_x
         * @param stop_x
         * @param inc_x
         * @param start_y
         * @param stop_y
         * @param inc_y
         * @param start_z
         * @param stop_z
         * @param inc_z
         * @return
         */
        static int parallel_for(FUNC_3D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t start_z, uint64_t stop_z, uint64_t inc_z, uint64_t numThreads = nd4j::Environment::getInstance()->maxThreads());

        /**
         *
         * @param function
         * @param numThreads
         * @return
         */
        static int parallel_do(FUNC_DO function, uint64_t numThreads = nd4j::Environment::getInstance()->maxThreads());
    };
}


#endif //SAMEDIFF_THREADS_H
