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

#ifndef DEV_TESTS_CONSTANTHELPER_H
#define DEV_TESTS_CONSTANTHELPER_H

#include <op_boilerplate.h>
#include <dll.h>
#include <pointercast.h>
#include <memory/Workspace.h>
#include <vector>
#include <map>
#include <mutex>
#include <array/ConstantDescriptor.h>
#include <array/ConstantDataBuffer.h>
#include <array/ConstantHolder.h>

namespace nd4j {
    class ND4J_EXPORT ConstantHelper {
    private:
        static ConstantHelper* _INSTANCE;
        ConstantHelper();

        std::vector<MAP_IMPL<ConstantDescriptor, ConstantHolder*>> _cache;

        // tracking of per-device constant memory buffers (CUDA only atm)
        std::vector<Nd4jPointer> _devicePointers;
        std::vector<Nd4jLong> _deviceOffsets;
        std::mutex _mutex;
        std::mutex _mutexHolder;

        std::vector<Nd4jLong> _counters;
    public:
        ~ConstantHelper() = default;

        static ConstantHelper* getInstance();
        static int getCurrentDevice();
        static int getNumberOfDevices();
        void* replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace = nullptr);

        ConstantDataBuffer* constantBuffer(const ConstantDescriptor &descriptor, nd4j::DataType dataType);

        Nd4jLong getCachedAmount(int deviceId);
    };
}

#endif //DEV_TESTS_CONSTANTHELPER_H
