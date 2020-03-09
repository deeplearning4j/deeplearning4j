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

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <loops/type_conversions.h>
#include <execution/Threads.h>
#include <chrono>
#include <execution/ThreadPool.h>

using namespace sd;
using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class ThreadsTests : public testing::Test {
public:
    ThreadsTests() {
        nd4j_printf("\n","");
    }
};

TEST_F(ThreadsTests, th_test_1) {
    ASSERT_EQ(1, ThreadsHelper::numberOfThreads(6, 1023));
    ASSERT_EQ(1, ThreadsHelper::numberOfThreads(6, 1024));
    ASSERT_EQ(1, ThreadsHelper::numberOfThreads(6, 1026));

    ASSERT_EQ(1, ThreadsHelper::numberOfThreads(6, 2043));
    ASSERT_EQ(2, ThreadsHelper::numberOfThreads(6, 2048));
}


TEST_F(ThreadsTests, th_test_2) {
    // in this case we'll get better split over second loop - exactly 32 elements per thread
    ASSERT_EQ(2, ThreadsHelper::pickLoop2d(32, 48, 1024));
    ASSERT_EQ(2, ThreadsHelper::pickLoop2d(6, 4, 16384));

    // in this case we'll get better split over first loop - 2 loops/2048 elements per thread
    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(32, 64, 1024));
    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(6, 6, 16384));

    // in this case none of loops are good enough, but second loop is too small for split
    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(6, 64, 32));

    // all loops are good enough, but we go with bigger one, since small
    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(2, 64, 32));

    // obviously split goes into second loop, to give 1024 elements per thread
    ASSERT_EQ(2, ThreadsHelper::pickLoop2d(2, 1, 2048));
}

TEST_F(ThreadsTests, th_test_3) {
    // typical conv cases
    ASSERT_EQ(1, ThreadsHelper::pickLoop3d(4, 32, 3, 128));
    ASSERT_EQ(2, ThreadsHelper::pickLoop3d(4, 1, 128, 64));
    ASSERT_EQ(3, ThreadsHelper::pickLoop3d(4, 1, 3, 128));

    // checking for optimal threads for conv inference
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads3d(6, 1, 3, 128));
    ASSERT_EQ(4, ThreadsHelper::numberOfThreads3d(4, 1, 3, 128));
    ASSERT_EQ(8, ThreadsHelper::numberOfThreads3d(8, 1, 3, 128));

    // checking for optimal threads for conv training
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads3d(6, 16, 3, 128));
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads3d(6, 8, 3, 128));


    ASSERT_EQ(6, ThreadsHelper::numberOfThreads3d(6, 8, 3, 64));
    ASSERT_EQ(1, ThreadsHelper::pickLoop3d(6, 8, 3, 64));
}

TEST_F(ThreadsTests, th_test_5) {
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads3d(6, 32, 112, 112));

    ASSERT_EQ(1, ThreadsHelper::pickLoop3d(6, 32, 112, 112));

    for (auto e = 0; e < 6; e++) {
        auto span = Span3::build(1, e, 6, 0, 32, 1, 0, 112, 1, 0, 112, 1);

        nd4j_printf("Span start: %lld; stop: %lld\n", span.startX(), span.stopX());
    }
}

TEST_F(ThreadsTests, th_test_4) {
    // typical conv cases
    ASSERT_EQ(2, ThreadsHelper::numberOfThreads2d(2, 32, 3));
    ASSERT_EQ(4, ThreadsHelper::numberOfThreads2d(4, 32, 3));
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads2d(6, 32, 1));
    ASSERT_EQ(8, ThreadsHelper::numberOfThreads2d(8, 16, 64));

    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(4, 32, 1));
    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(8, 19, 17));

    // primes edge cases
    ASSERT_EQ(6, ThreadsHelper::numberOfThreads2d(6, 19, 17));
    ASSERT_EQ(8, ThreadsHelper::numberOfThreads2d(8, 19, 17));

    ASSERT_EQ(1, ThreadsHelper::pickLoop2d(8, 19, 17));

    for (auto e = 0; e < 6; e++) {
        auto span = Span2::build(1, e, 6, 0, 19, 1, 0, 17, 1);

        nd4j_printf("Span start: %lld; stop: %lld\n", span.startX(), span.stopX());
    }

    nd4j_printf("-----------------------\n","");
    for (auto e = 0; e < 6; e++) {
        auto span = Span2::build(1, e, 6, 0, 32, 1, 0, 3, 1);

        nd4j_printf("Span start: %lld; stop: %lld\n", span.startX(), span.stopX());
    }
}


TEST_F(ThreadsTests, test_span_converage_1) {
    for (int b = 1; b <= 128; b++) {
        for (int c = 1; c <= 64; c++) {
            for (int t = 1; t <= 64; t++) {

                auto threads = ThreadsHelper::numberOfThreads2d(t, b, c);
                auto loop = ThreadsHelper::pickLoop2d(threads, b, c);

                if (t > 1 && threads == 1 && (b > 1 && c > 1)) {
                    nd4j_printf("Got 1 thread for [%i, %i] loop; initial max threads: %i\n", b, c, t)
                }

                auto sum = 0;
                for (auto a = 0; a < threads; a++) {
                    auto span = Span2::build(loop, a,threads, 0, b, 1, 0, c, 1);

                    if (loop == 1)
                        sum += span.stopX() - span.startX();
                    else if (loop == 2)
                        sum += span.stopY() - span.startY();
                    else
                        throw std::runtime_error("Bad loop!");
                }

                if (loop == 1)
                    ASSERT_EQ(b, sum);
                else
                    ASSERT_EQ(c, sum);
            }
        }
    }
}

TEST_F(ThreadsTests, validation_test_2d_1) {
    if (1 > 0)
        return;

    std::vector<int> threads({1, 2, 4, 6, 8, 12, 16, 20, 32, 48, 64});

    for (int e = 1; e < 1024; e++) {
        for (int i = 1; i <= 1024; i++ ) {
            for (auto t:threads) {
                std::atomic<int64_t> sum;
                sum.store(0);

                auto func = PRAGMA_THREADS_FOR_2D {
                    for (auto x = start_x; x < stop_x; x += inc_x) {
                        for (auto y = start_y; y < stop_y; y += inc_y) {
                            sum++;
                        }
                    }
                };

                sd::Threads::parallel_for(func, 0, e, 1, 0, i, 1, t, true);

                ASSERT_EQ(e * i, sum.load());
            }
        }

        nd4j_printf("Finished iteration %i\n", e);
    }
}

TEST_F(ThreadsTests, reduction_test_1) {

    auto func = PRAGMA_REDUCE_LONG {
        int64_t sum = 0;

        for (auto e = start; e < stop; e++) {
            sum++;
        };

        return sum;
    };

    auto sum = sd::Threads::parallel_long(func, LAMBDA_AL {return _old + _new;}, 0, 8192, 1, 4);
    ASSERT_EQ(8192, sum);
}

/*
TEST_F(ThreadsTests, basic_test_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto instance = sd::ThreadPool::getInstance();

    auto array = NDArrayFactory::create<float>('c', {512, 768});
    auto like = array.like();
    auto buffer = array.bufferAsT<float>();
    auto lbuffer = like.bufferAsT<float>();

    auto func = PRAGMA_THREADS_FOR {
        PRAGMA_OMP_SIMD
        for (uint64_t e = start; e < stop; e += increment) {
            buffer[e] += 1.0f;
        }
    };

    auto timeStartThreads = std::chrono::system_clock::now();
    sd::Threads::parallel_for(func, 0, array.lengthOf());
    auto timeEndThreads = std::chrono::system_clock::now();
    auto outerTimeThreads = std::chrono::duration_cast<std::chrono::microseconds> (timeEndThreads - timeStartThreads).count();

    auto timeStartOmp = std::chrono::system_clock::now();
    PRAGMA_OMP_PARALLEL_FOR_SIMD
    for (uint64_t e = 0; e < array.lengthOf(); e ++) {
        lbuffer[e] += 1.0f;
    }
    auto timeEndOmp = std::chrono::system_clock::now();
    auto outerTimeOmp = std::chrono::duration_cast<std::chrono::microseconds> (timeEndOmp - timeStartOmp).count();

    ASSERT_NEAR((float) array.lengthOf(), array.sumNumber().e<float>(0), 1e-5f);

    nd4j_printf("Threads time: %lld us; OMP time: %lld us; %p\n", outerTimeThreads, outerTimeOmp, instance)
}
 */