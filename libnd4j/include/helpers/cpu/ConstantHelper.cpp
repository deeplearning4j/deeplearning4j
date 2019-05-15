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
//  @author raver119@gmail.com
//

#include <ConstantHelper.h>
#include <cstring>

namespace nd4j {
    ConstantHelper::ConstantHelper() {
        //
    }

    ConstantHelper* ConstantHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new nd4j::ConstantHelper();

        return _INSTANCE;
    }

    void* ConstantHelper::replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace) {
        int8_t *ptr = nullptr;
        ALLOCATE(ptr, workspace, numBytes, int8_t);
        std::memcpy(ptr, src, numBytes);
        return ptr;
    }

    int ConstantHelper::getCurrentDevice() {
        return 0L;
    }

    int ConstantHelper::getNumberOfDevices() {
        return 1;
    }

    nd4j::ConstantHelper* nd4j::ConstantHelper::_INSTANCE = 0;
}