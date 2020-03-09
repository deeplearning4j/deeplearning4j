/*******************************************************************************
 * Copyright (c) 2019 Konduit
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
#include <system/openmp_pragmas.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <system/op_enums.h>

namespace samediff {
    class ND4J_EXPORT ThreadsHelper {
    public:
        static int numberOfThreads(int maxThreads, uint64_t numberOfElements);
        static int numberOfThreads2d(int maxThreads, uint64_t iters_x, uint64_t iters_y);
        static int numberOfThreads3d(int maxThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z);
        static int pickLoop2d(int numThreads, uint64_t iters_x, uint64_t iters_y);
        static int pickLoop3d(int numThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z);
    };

    class ND4J_EXPORT Span {
    private:
        int64_t _startX, _stopX, _incX;
    public:
        Span(int64_t start_x, int64_t stop_x, int64_t inc_x);
        ~Span() = default;

        int64_t startX() const;
        int64_t stopX() const;
        int64_t incX() const;

        static Span build(uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x);
    };

    class ND4J_EXPORT Span2 {
    private:
        int64_t _startX, _stopX, _incX;
        int64_t _startY, _stopY, _incY;
    public:
        Span2(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
        ~Span2() = default;

        int64_t startX() const;
        int64_t startY() const;

        int64_t stopX() const;
        int64_t stopY() const;

        int64_t incX() const;
        int64_t incY() const;

        static Span2 build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
    };

    class ND4J_EXPORT Span3 {
    private:
        int64_t _startX, _stopX, _incX;
        int64_t _startY, _stopY, _incY;
        int64_t _startZ, _stopZ, _incZ;
    public:
        Span3(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z);
        ~Span3() = default;

        int64_t startX() const;
        int64_t startY() const;
        int64_t startZ() const;

        int64_t stopX() const;
        int64_t stopY() const;
        int64_t stopZ() const;

        int64_t incX() const;
        int64_t incY() const;
        int64_t incZ() const;

        static Span3 build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z);
    };

    class ND4J_EXPORT Threads {
    public:
        /**
         * This function executes 1 dimensional loop for a given number of threads
         * PLEASE NOTE: this function can use smaller number of threads than requested.
         *
         * @param function
         * @param numThreads
         * @param start
         * @param stop
         * @param increment
         * @return
         */
        static int parallel_for(FUNC_1D function, int64_t start, int64_t stop, int64_t increment = 1, uint32_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This function executes 1 dimensional loop for a given number of threads
         *
         * @param function
         * @param start
         * @param stop
         * @param increment
         * @param numThreads
         * @return
         */
        static int parallel_tad(FUNC_1D function, int64_t start, int64_t stop, int64_t increment = 1, uint32_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This method will execute function splitting 2 nested loops space with multiple threads
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
        static int parallel_for(FUNC_2D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads(), bool debug = false);

        /**
         * This method will execute function splitting 3 nested loops space with multiple threads
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
        static int parallel_for(FUNC_3D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         *
         * @param function
         * @param numThreads
         * @return
         */
        static int parallel_do(FUNC_DO function, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        static int64_t parallel_long(FUNC_RL function, FUNC_AL aggregator, int64_t start, int64_t stop, int64_t increment = 1, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        static double parallel_double(FUNC_RD function, FUNC_AD aggregator, int64_t start, int64_t stop, int64_t increment = 1, uint64_t numThreads = sd::Environment::getInstance()->maxMasterThreads());

        /**
         * This method will execute function in parallel preserving the parts to be aligned increment size
         * PLEASE NOTE: this function can use smaller number of threads than requested.
         *
        */
        static int  parallel_aligned_increment(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, size_t type_size = sizeof(float), uint32_t req_numThreads = sd::Environment::getInstance()->maxMasterThreads());

    };
}


#endif //SAMEDIFF_THREADS_H
