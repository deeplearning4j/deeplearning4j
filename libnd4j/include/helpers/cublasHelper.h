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

#ifndef DEV_TESTS_CUBLASHELPER_H
#define DEV_TESTS_CUBLASHELPER_H

#include <dll.h>
#include <pointercast.h>
#include <vector>
#include <mutex>

namespace nd4j {
    class CublasHelper {
    private:
        static CublasHelper *_INSTANCE;
        static std::mutex _mutex;

        std::vector<void*> _cache;
        std::vector<void*> _solvers;

        CublasHelper();
        ~CublasHelper();
    public:
        static CublasHelper* getInstance();

        void* solver();

        void* handle();
        void* handle(int deviceId);
    };
}

#endif //DEV_TESTS_CUBLASHELPER_H
