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

#include "../cublasHelper.h"

namespace sd {
    static void* handle_() {
        return nullptr;
    }

    static void destroyHandle_(void* handle) {

    }

    CublasHelper::CublasHelper() {

    }

    CublasHelper::~CublasHelper() {

    }

    CublasHelper* CublasHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new sd::CublasHelper();

        return _INSTANCE;
    }

    void* CublasHelper::handle() {
        return nullptr;
    }

    void* CublasHelper::solver() {
        return nullptr;
    }

    void* CublasHelper::handle(int deviceId) {
        return nullptr;
    }


    sd::CublasHelper* sd::CublasHelper::_INSTANCE = 0;
}