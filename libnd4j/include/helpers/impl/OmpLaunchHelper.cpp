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
// Created by raver on 6/30/2018.
//

#include <helpers/OmpLaunchHelper.h>
#include <Environment.h>
#include <templatemath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nd4j {
    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N) {
        return OmpLaunchHelper::betterSpan(N, OmpLaunchHelper::betterThreads(N));
    }

    Nd4jLong OmpLaunchHelper::betterSpan(Nd4jLong N, Nd4jLong numThreads) {
        auto r = N % numThreads;
        auto t = N / numThreads;
        
        if (r == 0)
            return t;
        else {
            // fuck alignment
            return t + 1;
        }
    }

    int OmpLaunchHelper::betterThreads(Nd4jLong N) {
        #ifdef _OPENMP
            return betterThreads(N, omp_get_max_threads());
        #else
            return 1;
        #endif
    }

    int OmpLaunchHelper::betterThreads(Nd4jLong N, int maxThreads) {
        auto t = Environment::getInstance()->elementwiseThreshold();
        if (N < t)
            return 1;
        else {
            return static_cast<int>(nd4j::math::nd4j_min<Nd4jLong>(N / t, maxThreads));
        }
    }
}
