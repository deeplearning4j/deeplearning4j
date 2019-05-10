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
// Created by raver119 on 31.10.2017.
//

#include <helpers/logger.h>

namespace nd4j {


#ifdef __CUDACC__
    __host__
#endif
    void Logger::info(const char *format, ...) {
        va_list args;
        va_start(args, format);

        vprintf(format, args);

        va_end(args);

        fflush(stdout);
    }

#ifdef __CUDACC__
    __host__
#endif
     void Logger::printv(const char *format, std::vector<int>& vec) {
        printf("%s: {", format);
        for(int e = 0; e < vec.size(); e++) {
            auto v = vec[e];
            printf("%i", v);
            if (e < vec.size() - 1)
                printf(", ");
        }
        printf("}\n");
        fflush(stdout);
    }

    #ifdef __CUDACC__
    __host__
#endif
     void Logger::printv(const char *format, std::vector<Nd4jLong>& vec) {
        printf("%s: {", format);
        for(int e = 0; e < vec.size(); e++) {
            auto v = vec[e];
            printf("%lld", (long long) v);
            if (e < vec.size() - 1)
                printf(", ");
        }
        printf("}\n");
        fflush(stdout);
    }
}