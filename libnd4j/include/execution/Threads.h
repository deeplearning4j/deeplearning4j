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
         */
        static void parallel_for(FUNC_1D function, uint32_t numThreads, uint64_t start, uint64_t stop, uint64_t increment = 1);



    };
}


#endif //SAMEDIFF_THREADS_H
